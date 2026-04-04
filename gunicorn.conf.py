from app import post_fork

# Gunicorn will pick up the imported callable because the variable name matches
# the expected hook name.
