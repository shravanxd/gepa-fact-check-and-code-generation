"""
Code Executor Module
====================

Provides safe execution of generated Python code against unit tests.
Implements timeout handling, error capture, and detailed execution feedback.

This module follows the Single Responsibility Principle - it only handles
code execution and result capture.
"""

import ast
import sys
import traceback
import multiprocessing
from io import StringIO
from typing import Optional
from dataclasses import dataclass
from enum import Enum
from contextlib import redirect_stdout, redirect_stderr


class ExecutionStatus(Enum):
    """Enum representing possible execution outcomes"""
    PASSED = "passed"
    FAILED = "failed"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    IMPORT_ERROR = "import_error"


@dataclass
class ExecutionResult:
    """
    Data class holding the result of code execution.
    
    Attributes:
        status: The execution outcome status
        passed: Whether all tests passed
        error_type: Type of error if any (e.g., 'AssertionError', 'TypeError')
        error_message: Human-readable error message
        traceback: Full traceback string for debugging
        stdout: Captured standard output
        stderr: Captured standard error
        execution_time: Time taken to execute (in seconds)
    """
    status: ExecutionStatus
    passed: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    execution_time: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "status": self.status.value,
            "passed": self.passed,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "traceback": self.traceback,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "execution_time": self.execution_time
        }
    
    def get_feedback(self) -> str:
        """
        Generate human-readable feedback for reflection.
        This is used by GEPA's reflective loop.
        """
        if self.passed:
            return "PASSED: All tests passed successfully."
        
        feedback_parts = [f"FAILED: {self.status.value}"]
        
        if self.error_type:
            feedback_parts.append(f"Error Type: {self.error_type}")
        
        if self.error_message:
            feedback_parts.append(f"Error Message: {self.error_message}")
        
        if self.traceback:
            # Truncate traceback if too long
            tb = self.traceback
            if len(tb) > 1000:
                tb = tb[:1000] + "\n... (truncated)"
            feedback_parts.append(f"Traceback:\n{tb}")
        
        if self.stdout and self.stdout.strip():
            feedback_parts.append(f"Stdout:\n{self.stdout[:500]}")
        
        if self.stderr and self.stderr.strip():
            feedback_parts.append(f"Stderr:\n{self.stderr[:500]}")
        
        return "\n\n".join(feedback_parts)


class CodeValidator:
    """
    Validates Python code for syntax errors before execution.
    Implements Single Responsibility - only validates syntax.
    """
    
    @staticmethod
    def validate_syntax(code: str) -> tuple[bool, Optional[str]]:
        """
        Check if code has valid Python syntax.
        
        Args:
            code: Python code string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            error_msg = f"Line {e.lineno}: {e.msg}"
            return False, error_msg


class CodeExtractor:
    """
    Extracts Python code from LLM responses.
    Handles various formats like markdown code blocks.
    """
    
    @staticmethod
    def extract_code(response: str) -> str:
        """
        Extract Python code from LLM response.
        
        Handles:
        - ```python ... ``` blocks
        - ``` ... ``` blocks
        - Raw code without blocks
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Extracted Python code
        """
        response = response.strip()
        
        # Try to extract from ```python ... ``` block
        if "```python" in response:
            start = response.find("```python") + len("```python")
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        # Try to extract from ``` ... ``` block
        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                code = response[start:end].strip()
                # Skip language identifier if present on first line
                lines = code.split('\n')
                if lines and lines[0].strip() in ['python', 'py', '']:
                    code = '\n'.join(lines[1:])
                return code.strip()
        
        # Return as-is if no code blocks found
        return response


def _execute_code_in_process(code: str, test_code: str, result_queue: multiprocessing.Queue):
    """
    Worker function that executes code in a separate process.
    
    This function runs in an isolated process for safety and timeout handling.
    
    Args:
        code: The generated code to execute
        test_code: The test code to run against the generated code
        result_queue: Queue to put the result back to parent process
    """
    import time
    start_time = time.time()
    
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    
    try:
        # Combine generated code with test code
        full_code = f"{code}\n\n{test_code}"
        
        # Create a restricted global namespace
        exec_globals = {
            "__builtins__": __builtins__,
            "__name__": "__main__",
        }
        
        # Execute with stdout/stderr capture
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(full_code, exec_globals)
        
        execution_time = time.time() - start_time
        
        result = ExecutionResult(
            status=ExecutionStatus.PASSED,
            passed=True,
            stdout=stdout_capture.getvalue(),
            stderr=stderr_capture.getvalue(),
            execution_time=execution_time
        )
        
    except SyntaxError as e:
        execution_time = time.time() - start_time
        result = ExecutionResult(
            status=ExecutionStatus.SYNTAX_ERROR,
            passed=False,
            error_type="SyntaxError",
            error_message=f"Line {e.lineno}: {e.msg}",
            traceback=traceback.format_exc(),
            stdout=stdout_capture.getvalue(),
            stderr=stderr_capture.getvalue(),
            execution_time=execution_time
        )
        
    except AssertionError as e:
        execution_time = time.time() - start_time
        result = ExecutionResult(
            status=ExecutionStatus.FAILED,
            passed=False,
            error_type="AssertionError",
            error_message=str(e) if str(e) else "Assertion failed",
            traceback=traceback.format_exc(),
            stdout=stdout_capture.getvalue(),
            stderr=stderr_capture.getvalue(),
            execution_time=execution_time
        )
        
    except ImportError as e:
        execution_time = time.time() - start_time
        result = ExecutionResult(
            status=ExecutionStatus.IMPORT_ERROR,
            passed=False,
            error_type="ImportError",
            error_message=str(e),
            traceback=traceback.format_exc(),
            stdout=stdout_capture.getvalue(),
            stderr=stderr_capture.getvalue(),
            execution_time=execution_time
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        result = ExecutionResult(
            status=ExecutionStatus.RUNTIME_ERROR,
            passed=False,
            error_type=type(e).__name__,
            error_message=str(e),
            traceback=traceback.format_exc(),
            stdout=stdout_capture.getvalue(),
            stderr=stderr_capture.getvalue(),
            execution_time=execution_time
        )
    
    result_queue.put(result)


class CodeExecutor:
    """
    Main executor class that safely runs Python code against tests.
    
    Uses multiprocessing for isolation and timeout handling.
    Follows Interface Segregation - provides a clean execute() interface.
    """
    
    def __init__(self, timeout: float = 5.0):
        """
        Initialize the executor.
        
        Args:
            timeout: Maximum execution time in seconds (default: 5.0)
        """
        self.timeout = timeout
        self.validator = CodeValidator()
        self.extractor = CodeExtractor()
    
    def execute(
        self, 
        generated_code: str, 
        test_code: str,
        extract_from_response: bool = True
    ) -> ExecutionResult:
        """
        Execute generated code against test code.
        
        Args:
            generated_code: The code generated by the LLM
            test_code: The unit test code from HumanEval
            extract_from_response: Whether to extract code from markdown blocks
            
        Returns:
            ExecutionResult with pass/fail status and feedback
        """
        # Step 1: Extract code if needed
        if extract_from_response:
            code = self.extractor.extract_code(generated_code)
        else:
            code = generated_code
        
        # Step 2: Validate syntax before execution
        is_valid, syntax_error = self.validator.validate_syntax(code)
        if not is_valid:
            return ExecutionResult(
                status=ExecutionStatus.SYNTAX_ERROR,
                passed=False,
                error_type="SyntaxError",
                error_message=syntax_error,
                traceback=None
            )
        
        # Step 3: Execute in isolated process with timeout
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_execute_code_in_process,
            args=(code, test_code, result_queue)
        )
        
        process.start()
        process.join(timeout=self.timeout)
        
        if process.is_alive():
            # Timeout occurred
            process.terminate()
            process.join(timeout=1.0)
            
            # Force kill if still alive
            if process.is_alive():
                process.kill()
                process.join()
            
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                passed=False,
                error_type="TimeoutError",
                error_message=f"Execution exceeded {self.timeout} seconds",
                execution_time=self.timeout
            )
        
        # Get result from queue
        try:
            result = result_queue.get_nowait()
            return result
        except Exception:
            return ExecutionResult(
                status=ExecutionStatus.RUNTIME_ERROR,
                passed=False,
                error_type="UnknownError",
                error_message="Failed to retrieve execution result"
            )
    
    def execute_with_entry_point(
        self,
        generated_code: str,
        test_code: str,
        entry_point: str,
        extract_from_response: bool = True
    ) -> ExecutionResult:
        """
        Execute code with a specific entry point function.
        
        This is useful for HumanEval where we need to call a specific function.
        
        Args:
            generated_code: The code generated by the LLM
            test_code: The unit test code
            entry_point: The function name to test
            extract_from_response: Whether to extract code from markdown blocks
            
        Returns:
            ExecutionResult with pass/fail status and feedback
        """
        # Extract code if needed
        if extract_from_response:
            code = self.extractor.extract_code(generated_code)
        else:
            code = generated_code
        
        # Check if entry point exists in the code
        if entry_point not in code:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                passed=False,
                error_type="MissingFunction",
                error_message=f"Function '{entry_point}' not found in generated code"
            )
        
        return self.execute(code, test_code, extract_from_response=False)


# Convenience function for simple usage
def run_code_with_tests(
    generated_code: str,
    test_code: str,
    timeout: float = 5.0
) -> ExecutionResult:
    """
    Convenience function to run code against tests.
    
    Args:
        generated_code: The code to test
        test_code: The test code to run
        timeout: Maximum execution time
        
    Returns:
        ExecutionResult
    """
    executor = CodeExecutor(timeout=timeout)
    return executor.execute(generated_code, test_code)
