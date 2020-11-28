

class Strategy(object):
    def __init__(
            self, intrin_match_result, transform_steps, schedules):
        self.intrin_match_result = intrin_match_result
        self.transform_steps = transform_steps
        self.schedules = schedules

    def get_intrin_match_result(self):
        return self.intrin_match_result

    def get_transform_steps(self):
        return self.transform_steps

    def get_schedules(self):
        return self.schedules
