def get_inhaler():
    class MockInhaler:
        def inhale(self, *args, **kwargs):
            return {"status": "ok"}
    return MockInhaler()
