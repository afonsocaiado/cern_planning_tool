from flask import jsonify, make_response

def bad_request(error):
    """Handles 400 Bad Request errors.

    Args:
        error: The error object.

    Returns:
        A JSON object containing the error message and an HTTP 400 status code.
    """

    return make_response(jsonify({"error": error.description}), 400)

def internal_server_error(error):
    """Handles 500 Internal Server Error errors.

    Args:
        error: The error object.

    Returns:
        A JSON object containing the error message and an HTTP 500 status code.
    """

    return make_response(jsonify({"error": "Internal Server Error"}), 500)

def register_error_handlers(app):
    """Registers error handlers for the given Flask app.

    Args:
        app (Flask): The Flask app instance to register error handlers for.
    """

    app.register_error_handler(400, bad_request)
    app.register_error_handler(500, internal_server_error)