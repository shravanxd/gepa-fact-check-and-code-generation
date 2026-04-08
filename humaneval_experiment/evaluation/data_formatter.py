"""
Data Formatter Module for HumanEval
====================================

Handles data format conversion from HumanEval dataset to GEPA format.
Follows Single Responsibility Principle - only handles data formatting.
"""

from typing import Dict, Any


class DataFormatter:
    """
    Handles data format conversion for HumanEval dataset.
    
    Converts HumanEval examples to a standardized GEPA format that
    can be used by the HumanEvalAdapter.
    """
    
    @staticmethod
    def humaneval_to_gepa_format(example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a HumanEval example to GEPA format.
        
        HumanEval dataset structure:
        - task_id: str (e.g., "HumanEval/0")
        - prompt: str (function signature + docstring)
        - canonical_solution: str (reference solution)
        - test: str (unit test code)
        - entry_point: str (function name to test)
        
        GEPA format:
        - input: str (the prompt for the LLM)
        - answer: str (canonical solution for reference)
        - additional_context: dict (metadata including tests)
        
        Args:
            example: HumanEval dataset example
            
        Returns:
            Dict in GEPA format
        """
        # Extract fields from HumanEval format
        task_id = example.get('task_id', '')
        prompt = example.get('prompt', '')
        canonical_solution = example.get('canonical_solution', '')
        test = example.get('test', '')
        entry_point = example.get('entry_point', '')
        
        # Build the input prompt
        # The prompt from HumanEval already contains function signature + docstring
        input_text = prompt
        
        return {
            "input": input_text,
            "answer": canonical_solution,  # Reference solution (not used for scoring)
            "additional_context": {
                "task_id": task_id,
                "entry_point": entry_point,
                "test": test,
            }
        }
    
    @staticmethod
    def format_problem_display(example: Dict[str, Any]) -> str:
        """
        Format a problem for human-readable display.
        
        Useful for debugging and logging.
        
        Args:
            example: GEPA-formatted example
            
        Returns:
            Formatted string for display
        """
        task_id = example.get('additional_context', {}).get('task_id', 'Unknown')
        entry_point = example.get('additional_context', {}).get('entry_point', 'Unknown')
        input_text = example.get('input', '')
        
        # Truncate if too long
        if len(input_text) > 500:
            input_preview = input_text[:500] + "..."
        else:
            input_preview = input_text
        
        return f"""
{'='*60}
Task ID: {task_id}
Entry Point: {entry_point}
{'='*60}
{input_preview}
{'='*60}
"""
    
    @staticmethod
    def extract_function_signature(prompt: str) -> str:
        """
        Extract the function signature from a HumanEval prompt.
        
        Args:
            prompt: The HumanEval prompt string
            
        Returns:
            The function signature line
        """
        lines = prompt.strip().split('\n')
        for line in lines:
            if line.strip().startswith('def '):
                return line.strip()
        return ""
    
    @staticmethod
    def extract_docstring(prompt: str) -> str:
        """
        Extract the docstring from a HumanEval prompt.
        
        Args:
            prompt: The HumanEval prompt string
            
        Returns:
            The docstring content
        """
        # Find triple quotes
        start_markers = ['"""', "'''"]
        
        for marker in start_markers:
            if marker in prompt:
                start = prompt.find(marker)
                end = prompt.find(marker, start + 3)
                if end != -1:
                    return prompt[start:end + 3]
        
        return ""
