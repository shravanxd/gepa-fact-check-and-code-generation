"""
Custom GEPA Adapter for HumanEval Dataset
==========================================

This adapter provides custom evaluation and reflection for the HumanEval
code generation task. It handles:
1. Code generation via LLM
2. Execution-based evaluation using unit tests
3. Rich feedback generation for GEPA's reflective loop

Follows SOLID principles:
- Single Responsibility: Adapter only handles GEPA integration
- Open/Closed: Extendable for different code tasks
- Dependency Inversion: Depends on abstractions (CodeExecutor interface)
"""

from typing import Any, Callable, TypedDict, Optional
from dataclasses import dataclass

from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from code_executor import CodeExecutor, ExecutionResult, ExecutionStatus, CodeExtractor


# ============================================================================
# Type Definitions
# ============================================================================

class HumanEvalDataInst(TypedDict):
    """
    Data instance for a HumanEval problem.
    
    Attributes:
        input: The prompt (function signature + docstring)
        answer: The canonical solution (for reference, not used in evaluation)
        additional_context: Contains task_id, entry_point, and test code
    """
    input: str  # Function signature + docstring
    answer: str  # Canonical solution (reference only)
    additional_context: dict[str, Any]  # task_id, entry_point, test


class HumanEvalTrajectory(TypedDict):
    """
    Trajectory data capturing the full execution trace.
    
    Used for reflection to understand what went wrong.
    """
    data: HumanEvalDataInst
    full_response: str  # Raw LLM output
    extracted_code: str  # Code after extraction
    execution_result: dict  # ExecutionResult as dict


class HumanEvalRolloutOutput(TypedDict):
    """
    Output from a single rollout/evaluation.
    """
    full_response: str
    extracted_code: str
    passed: bool
    error_type: Optional[str]
    error_message: Optional[str]


# ============================================================================
# Adapter Implementation
# ============================================================================

class HumanEvalAdapter(GEPAAdapter[HumanEvalDataInst, HumanEvalTrajectory, HumanEvalRolloutOutput]):
    """
    Custom GEPA adapter for HumanEval code generation task.
    
    This adapter:
    1. Generates code using an LLM given a function signature + docstring
    2. Executes the generated code against unit tests
    3. Provides detailed feedback for reflection including tracebacks
    
    The key difference from HoVerAdapter is that evaluation is execution-based,
    not string matching. This enables richer feedback for the reflective loop.
    """
    
    def __init__(
        self,
        model: str | Callable,
        timeout: float = 5.0,
        failure_score: float = 0.0,
        max_litellm_workers: int = 10,
        litellm_batch_completion_kwargs: dict[str, Any] = None,
    ):
        """
        Initialize the HumanEval adapter.
        
        Args:
            model: Model name for litellm or a callable that takes messages
            timeout: Execution timeout in seconds for each test
            failure_score: Score to assign on complete failure (default: 0.0)
            max_litellm_workers: Max parallel workers for batch completion
            litellm_batch_completion_kwargs: Additional kwargs for litellm
        """
        if isinstance(model, str):
            import litellm
            self.litellm = litellm
        
        self.model = model
        self.timeout = timeout
        self.failure_score = failure_score
        self.max_litellm_workers = max_litellm_workers
        self.litellm_batch_completion_kwargs = litellm_batch_completion_kwargs or {}
        
        # Initialize executor and extractor
        self.executor = CodeExecutor(timeout=timeout)
        self.extractor = CodeExtractor()
    
    def _build_user_prompt(self, data: HumanEvalDataInst) -> str:
        """
        Build the user prompt from a HumanEval data instance.
        
        Handles few-shot examples if present.
        
        Args:
            data: The HumanEval problem instance
            
        Returns:
            Formatted user prompt string
        """
        user_content = data['input']
        
        # Handle few-shot examples if present
        if isinstance(data, dict) and data.get("few_shot"):
            few = data.get("few_shot")
            try:
                if isinstance(few, list):
                    examples_text = []
                    for i_ex, ex in enumerate(few, start=1):
                        inp = ex.get("input") if isinstance(ex, dict) else str(ex)
                        sol = ex.get("solution") if isinstance(ex, dict) else ""
                        examples_text.append(f"Example {i_ex}:\n{inp}\n\nSolution:\n{sol}")
                    few_text = "\n\n---\n\n".join(examples_text)
                elif isinstance(few, dict) and "raw" in few:
                    few_text = few["raw"]
                else:
                    few_text = str(few)
            except Exception:
                few_text = str(few)
            
            user_content = f"Here are some examples:\n\n{few_text}\n\n---\n\nNow solve this:\n\n{user_content}"
        
        return user_content
    
    def _call_llm(self, messages_batch: list[list[dict]]) -> list[str]:
        """
        Call the LLM with a batch of message lists.
        
        Args:
            messages_batch: List of message lists, each for one problem
            
        Returns:
            List of response strings
        """
        if isinstance(self.model, str):
            raw_responses = self.litellm.batch_completion(
                model=self.model,
                messages=messages_batch,
                max_workers=self.max_litellm_workers,
                **self.litellm_batch_completion_kwargs
            )
            
            responses = []
            for resp in raw_responses:
                if hasattr(resp, 'choices') and len(resp.choices) > 0:
                    responses.append(resp.choices[0].message.content.strip())
                else:
                    print(f"Warning: LLM call failed with: {resp}")
                    responses.append("")
            return responses
        else:
            # Callable model
            return [self.model(messages) for messages in messages_batch]
    
    def evaluate(
        self,
        batch: list[HumanEvalDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[HumanEvalTrajectory, HumanEvalRolloutOutput]:
        """
        Evaluate a candidate prompt on a batch of HumanEval problems.
        
        This method:
        1. Generates code for each problem using the LLM
        2. Executes each generated code against its unit tests
        3. Returns pass/fail scores and optional trajectories for reflection
        
        Args:
            batch: List of HumanEval problems
            candidate: Dict with 'system_prompt' key
            capture_traces: Whether to capture detailed trajectories
            
        Returns:
            EvaluationBatch with outputs, scores, and optionally trajectories
        """
        outputs: list[HumanEvalRolloutOutput] = []
        scores: list[float] = []
        trajectories: list[HumanEvalTrajectory] | None = [] if capture_traces else None
        
        system_content = candidate['system_prompt']
        
        # Step 1: Prepare batch requests for LLM
        messages_batch = []
        for data in batch:
            user_content = self._build_user_prompt(data)
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            messages_batch.append(messages)
        
        # Step 2: Get LLM responses
        try:
            responses = self._call_llm(messages_batch)
        except Exception as e:
            print(f"Error during LLM call: {e}")
            # Return failure for all examples
            outputs = [
                {
                    "full_response": "",
                    "extracted_code": "",
                    "passed": False,
                    "error_type": "LLMError",
                    "error_message": str(e)
                }
                for _ in batch
            ]
            scores = [self.failure_score for _ in batch]
            trajectories_out = None if not capture_traces else [
                {
                    "data": data,
                    "full_response": "",
                    "extracted_code": "",
                    "execution_result": {"status": "llm_error", "passed": False}
                }
                for data in batch
            ]
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories_out)
        
        # Step 3: Execute each generated code and evaluate
        for data, response in zip(batch, responses, strict=False):
            # Extract code from response
            extracted_code = self.extractor.extract_code(response)
            
            # Get test code and entry point
            additional_ctx = data.get('additional_context', {})
            test_code = additional_ctx.get('test', '')
            entry_point = additional_ctx.get('entry_point', '')
            
            # Execute code against tests
            if test_code:
                exec_result = self.executor.execute_with_entry_point(
                    generated_code=extracted_code,
                    test_code=test_code,
                    entry_point=entry_point,
                    extract_from_response=False  # Already extracted
                )
            else:
                # No test code available - can't evaluate
                exec_result = ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    passed=False,
                    error_type="NoTestCode",
                    error_message="No test code provided for this problem"
                )
            
            # Score: 1.0 if passed, 0.0 otherwise
            score = 1.0 if exec_result.passed else 0.0
            
            output: HumanEvalRolloutOutput = {
                "full_response": response,
                "extracted_code": extracted_code,
                "passed": exec_result.passed,
                "error_type": exec_result.error_type,
                "error_message": exec_result.error_message
            }
            
            outputs.append(output)
            scores.append(score)
            
            if capture_traces:
                trajectory: HumanEvalTrajectory = {
                    "data": data,
                    "full_response": response,
                    "extracted_code": extracted_code,
                    "execution_result": exec_result.to_dict()
                }
                trajectories.append(trajectory)
        
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)
    
    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[HumanEvalTrajectory, HumanEvalRolloutOutput],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Build reflective dataset for prompt improvement.
        
        This is where HumanEval shines for GEPA - we can provide rich feedback
        including:
        - Syntax errors with line numbers
        - Runtime errors with tracebacks
        - Assertion failures with test case details
        - Timeout information
        
        Args:
            candidate: Current candidate being evaluated
            eval_batch: Results from evaluate() with trajectories
            components_to_update: Components to generate reflection data for
            
        Returns:
            Dict mapping component names to lists of reflection examples
        """
        ret_d: dict[str, list[dict[str, Any]]] = {}
        
        assert len(components_to_update) == 1, "HumanEval adapter currently supports single component"
        comp = components_to_update[0]
        
        items: list[dict[str, Any]] = []
        trace_instances = list(zip(
            eval_batch.trajectories, 
            eval_batch.scores, 
            eval_batch.outputs, 
            strict=False
        ))
        
        for traj, score, output in trace_instances:
            data = traj['data']
            exec_result_dict = traj['execution_result']
            
            # Build status string
            status = "PASSED" if score > 0.0 else "FAILED"
            
            # Get problem metadata
            task_id = data.get('additional_context', {}).get('task_id', 'unknown')
            entry_point = data.get('additional_context', {}).get('entry_point', 'unknown')
            
            # Build detailed feedback
            feedback = self._build_feedback(
                status=status,
                task_id=task_id,
                entry_point=entry_point,
                prompt=data['input'],
                generated_code=traj['extracted_code'],
                exec_result=exec_result_dict
            )
            
            # Create reflection item with structured data
            reflection_item = {
                "Status": status,
                "Task ID": task_id,
                "Entry Point": entry_point,
                "Problem": data['input'],
                "Generated Code": traj['extracted_code'],
                "Execution Status": exec_result_dict.get('status', 'unknown'),
                "Error Type": exec_result_dict.get('error_type'),
                "Error Message": exec_result_dict.get('error_message'),
                "Traceback": exec_result_dict.get('traceback'),
                "Feedback": feedback,
                "Score": score
            }
            
            items.append(reflection_item)
        
        ret_d[comp] = items
        
        if len(items) == 0:
            raise Exception("No valid predictions found for any module.")
        
        return ret_d
    
    def _build_feedback(
        self,
        status: str,
        task_id: str,
        entry_point: str,
        prompt: str,
        generated_code: str,
        exec_result: dict
    ) -> str:
        """
        Build detailed feedback string for reflection.
        
        This feedback helps the reflective LLM understand what went wrong
        and how to improve the prompt.
        
        Args:
            status: "PASSED" or "FAILED"
            task_id: HumanEval task ID
            entry_point: Function name being tested
            prompt: Original problem prompt
            generated_code: Code generated by the LLM
            exec_result: Execution result dictionary
            
        Returns:
            Formatted feedback string
        """
        if status == "PASSED":
            return f"""PASSED: Task {task_id}

The generated code for function `{entry_point}` passed all tests.

Generated Code:
```python
{generated_code}
```

Notes for prompt optimization:
- This is a successful case - the prompt worked well for this type of problem
- Consider what aspects of the prompt helped produce correct code"""
        
        # Build failure feedback with rich details
        exec_status = exec_result.get('status', 'unknown')
        error_type = exec_result.get('error_type', 'Unknown')
        error_message = exec_result.get('error_message', 'No message')
        traceback_str = exec_result.get('traceback', '')
        
        feedback_parts = [
            f"FAILED: Task {task_id}",
            f"Function: `{entry_point}`",
            f"Execution Status: {exec_status}",
            f"Error Type: {error_type}",
            f"Error Message: {error_message}",
        ]
        
        # Add problem context (truncated if too long)
        prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
        feedback_parts.append(f"\nProblem:\n{prompt_preview}")
        
        # Add generated code
        code_preview = generated_code[:800] + "..." if len(generated_code) > 800 else generated_code
        feedback_parts.append(f"\nGenerated Code:\n```python\n{code_preview}\n```")
        
        # Add traceback if available
        if traceback_str:
            tb_preview = traceback_str[:600] + "..." if len(traceback_str) > 600 else traceback_str
            feedback_parts.append(f"\nTraceback:\n{tb_preview}")
        
        # Add specific guidance based on error type
        guidance = self._get_error_guidance(exec_status, error_type)
        feedback_parts.append(f"\nGuidance:\n{guidance}")
        
        return "\n".join(feedback_parts)
    
    def _get_error_guidance(self, exec_status: str, error_type: str) -> str:
        """
        Get specific guidance based on error type.
        
        This helps the reflective LLM understand common issues.
        
        Args:
            exec_status: Execution status string
            error_type: Type of error encountered
            
        Returns:
            Guidance string
        """
        guidance_map = {
            "syntax_error": (
                "- The code has syntax errors and couldn't be parsed\n"
                "- Check for missing colons, parentheses, or indentation issues\n"
                "- Ensure the prompt asks for valid Python syntax"
            ),
            "timeout": (
                "- The code took too long to execute (possible infinite loop)\n"
                "- Check for while loops without proper termination conditions\n"
                "- The prompt should emphasize efficient solutions"
            ),
            "AssertionError": (
                "- The code ran but produced wrong output for test cases\n"
                "- The logic is incorrect - review the algorithm\n"
                "- Edge cases may not be handled properly"
            ),
            "TypeError": (
                "- Type mismatch occurred during execution\n"
                "- Check function parameters and return types\n"
                "- The prompt should clarify expected input/output types"
            ),
            "IndexError": (
                "- Array/list index out of bounds\n"
                "- Check loop bounds and edge cases (empty lists, etc.)\n"
                "- The prompt should emphasize handling edge cases"
            ),
            "KeyError": (
                "- Dictionary key not found\n"
                "- Check for missing dictionary entries\n"
                "- Consider using .get() with defaults"
            ),
            "ValueError": (
                "- Invalid value passed to a function\n"
                "- Check input validation and conversions\n"
                "- The prompt should mention input constraints"
            ),
            "MissingFunction": (
                "- The required function was not found in the generated code\n"
                "- The prompt should clearly specify the function name to implement\n"
                "- Ensure the model outputs the complete function definition"
            ),
        }
        
        # Check for matching guidance
        for key, guidance in guidance_map.items():
            if key.lower() in exec_status.lower() or key.lower() in error_type.lower():
                return guidance
        
        # Default guidance
        return (
            "- Review the error message and traceback for specific issues\n"
            "- Consider what prompt modifications could prevent this error\n"
            "- Focus on clarity and completeness in the prompt"
        )
