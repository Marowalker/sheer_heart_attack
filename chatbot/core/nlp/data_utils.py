class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
        ERROR: Unable to locate file {}.

        FIX: Have you tried running python train first?
        It will build your prediction models.""".format(filename)

        super(MyIOError, self).__init__(message)


class MyTypeError(Exception):
    def __init__(self):
        # custom error message
        message = """
        ERROR: One of the message that you tried to send is of the wrong input type.

        FIX: Have you tried following the bot's request?
        It will have a specific request for each type of input."""
        super(MyTypeError, self).__init__(message)
