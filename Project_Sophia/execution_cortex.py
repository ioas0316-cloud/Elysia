"""
Execution Cortex for Elysia.

This module acts as the 'will' of the AI, responsible for taking a concrete
plan (a sequence of tool calls) and executing it step-by-step. It manages the
context of the execution, passing the output of one step as the input to the next,
and asks for help when it encounters problems.
"""
import logging
import re
from .tool_executor import ToolExecutor

# --- Logging Configuration ---
logger = logging.getLogger(__name__)
# --- End Logging Configuration ---

class ExecutionCortex:
    """
    Executes a plan composed of a sequence of tool calls and asks for help
    when execution fails.
    """
    def __init__(self):
        self.tool_executor = ToolExecutor()

    def execute_plan(self, plan: list[dict]) -> str:
        """
        Executes a plan, managing the context between steps.

        Args:
            plan: A list of dictionaries, where each represents a tool call.

        Returns:
            A string representing the final result or a request for help.
        """
        logger.info(f"Executing plan with {len(plan)} steps.")

        execution_context = {}

        for i, step in enumerate(plan):
            tool_name = step.get("tool_name")
            parameters = step.get("parameters", {})

            logger.info(f"Executing step {i+1}: {tool_name} with params {parameters}")

            try:
                resolved_params = self._resolve_parameters(parameters, execution_context)

                result = self.tool_executor.execute(tool_name, resolved_params)

                step_output_key = f"step_{i+1}_output"
                execution_context[step_output_key] = result
                logger.info(f"Step {i+1} executed successfully. Result stored as '{step_output_key}'.")

            except ValueError as e:
                # Error resolving parameters, ask for the missing information
                logger.warning(f"Parameter resolution failed for step {i+1}: {e}")
                missing_info_match = re.search(r"'(.*?)'", str(e))
                missing_info = missing_info_match.group(1) if missing_info_match else "알 수 없는 정보"
                return f"계획을 실행하는 데 '{missing_info}' 정보가 부족합니다. 알려주시겠어요?"

            except FileNotFoundError as e:
                # Tool failed because a file was not found, ask for a new path
                logger.warning(f"File not found during step {i+1} ({tool_name}): {e}")
                filepath_match = re.search(r"'(.*?)'", str(e))
                filepath = filepath_match.group(1) if filepath_match else "알 수 없는"
                return f"경로 '{filepath}'에서 파일을 찾을 수 없습니다. 다른 경로를 시도해볼까요?"

            except NotImplementedError as e:
                # Tool is not fully implemented
                logger.error(f"Execution failed at step {i+1}: {e}")
                return f"'{tool_name}' 도구를 어떻게 사용해야 할지 아직 배우지 못했습니다. 다른 방법을 제안해주시겠어요?"

            except Exception as e:
                # Generic error, ask for general help
                logger.error(f"An unexpected error occurred at step {i+1} ({tool_name}): {e}", exc_info=True)
                return f"계획 실행 중 예상치 못한 문제에 부딪혔습니다. '{e}' 문제 해결을 도와주시겠어요?"

        final_result_key = f"step_{len(plan)}_output"
        final_result = execution_context.get(final_result_key, "계획을 성공적으로 완료했지만, 특별한 결과는 없습니다.")

        # Format the final result for the user
        return f"목표를 성공적으로 달성했습니다. 최종 결과: {final_result}"

    def _resolve_parameters(self, parameters: dict, context: dict) -> dict:
        """
        Resolves placeholders (e.g., '<step_1_output>') in parameters with
        actual values from the execution context.
        """
        resolved = {}
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith('<') and value.endswith('>'):
                context_key = value[1:-1]
                if context_key in context:
                    resolved[key] = context[context_key]
                else:
                    raise ValueError(f"Placeholder '{value}' could not be resolved from context.")
            else:
                resolved[key] = value
        return resolved
