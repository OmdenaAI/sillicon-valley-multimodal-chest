import sys

def error_message_detail(error, error_detail: sys):
    """
    Generates a detailed error message with file name, line number, and error message.

    Args:
        error: The error object or message.
        error_detail (sys): The sys object containing the traceback information.

    Returns:
        str: The error message with file name, line number, and error message.
    """
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_type = type(error).__name__

    error_message = "An error of type [{0}] occurred in the file [{1}] at line number [{2}]. Error message: {3}".format(
        error_type, file_name, line_number, str(error)
    )
    return error_message


class CustomException(Exception):
    """
    Custom exception class that inherits from the base Exception class.

    Args:
        error_message (str): The error message in string format.
        error_detail (sys): The sys object containing the traceback information.
    """

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)

        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        """
        Returns the error message as a string representation of the exception.

        Returns:
            str: The error message.
        """
        return self.error_message