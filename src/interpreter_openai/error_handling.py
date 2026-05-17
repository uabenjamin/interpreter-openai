from __future__ import annotations


class UserFacingError(RuntimeError):
    """An error message that should be shown directly to the user."""


def classify_openai_error(exc: Exception) -> str | None:
    status_code = getattr(exc, "status_code", None)
    message = getattr(exc, "message", None)
    if message is None:
        if exc.args:
            message = str(exc.args[0])
        else:
            message = str(exc)

    if status_code == 401:
        return (
            "OpenAI authentication failed. Check OPENAI_API_KEY and confirm the "
            "key is active."
        )
    if status_code == 403:
        return (
            "OpenAI access was denied. Check project-level access, model access, "
            "and billing."
        )
    if status_code == 404:
        return (
            "An OpenAI model or endpoint was not found. Check the configured model "
            "names and your account's model access."
        )
    if status_code == 429:
        return (
            "OpenAI rate limit or quota was hit. Check usage tier, quota, and "
            "project billing."
        )
    if status_code is not None and status_code >= 500:
        return f"OpenAI returned a server error ({status_code}). {message}"

    if "api key" in message.lower() and "openai" in message.lower():
        return (
            "OpenAI authentication failed. Check OPENAI_API_KEY and confirm the "
            "key is active."
        )
    if "beta_api_shape_disabled" in message.lower():
        return (
            "This app hit the removed OpenAI Realtime beta interface. Update to the "
            "latest interpreter-openai code that uses the GA Realtime session shape."
        )
    return None
