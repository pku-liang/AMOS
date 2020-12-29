from .auto_schedule import set_interpret, AutoScheduleMultiGraphDispatch
from .measure import set_evaluate_performance, start_evaluate, stop_evaluate, \
                     evaluate_function_for, auto_tensorize_for, start_tensorize, \
                     stop_tensorize
from .cost_model import set_query_cost_model