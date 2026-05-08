from langchain_core.tools import tool
import io
import contextlib

@tool
def execute_python_code(code: str) -> str:
    """Execute python code in a sandbox and return the stdout output."""
    # For local development we use a simple exec. 
    # CAUTION: Exec is not secure for production. Replace with docker-in-docker or restricted env for real prod.
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {})
        result = output.getvalue()
        return result if result else "Code executed successfully with no output."
    except Exception as e:
        return f"Error executing code: {str(e)}"
