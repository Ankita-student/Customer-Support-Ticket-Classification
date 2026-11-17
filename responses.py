"""Optional bonus task - automatic customer response (rule-based).

No external API, no API keys: every predicted category maps to a ready-made
reply template. In a real company this could be replaced by an LLM call.
"""

RESPONSES = {
    "Login Issue": (
        "Please use the 'Forgot Password' option on the login page to reset your password. "
        "If your account is locked, wait 15 minutes and try again. If the problem continues, "
        "please contact the support team and we will unlock your account."
    ),
    "Application Error": (
        "We are sorry about the error. Please refresh the page, clear your browser cache and try "
        "again. If it still fails, reply with a screenshot and the exact error message so our "
        "technical team can investigate."
    ),
    "Report": (
        "Thank you for your request. Please tell us the report type, date range and format "
        "(PDF/Excel/CSV) you need. Our team will generate it and share it with you shortly."
    ),
    "Account Update": (
        "To update your account details, go to Profile > Settings and edit the required field. "
        "For security reasons, changes to a mobile number or email may need OTP verification. "
        "If you need help, our team can complete the update after verifying your identity."
    ),
    "Performance": (
        "We are sorry for the slowness. Please check your internet connection, close unused tabs "
        "and try again. Our team has been notified of the performance issue and is looking into it."
    ),
    "Billing Issue": (
        "We apologise for the billing problem. Please share your invoice number and the payment "
        "date. Our billing team will review the charge and, if applicable, process a refund or "
        "correction within 5-7 working days."
    ),
}

DEFAULT_RESPONSE = (
    "Thank you for contacting support. Your request has been forwarded to our team, "
    "who will get back to you shortly."
)


def get_response(category):
    return RESPONSES.get(category, DEFAULT_RESPONSE)
