"""
Rate limiting utilities for Flask application
Provides simple in-memory rate limiting to prevent abuse
"""

import time
from functools import wraps
from collections import defaultdict, deque
from flask import request, jsonify, current_app
from flask_login import current_user


class RateLimiter:
    """Simple in-memory rate limiter using sliding window"""

    def __init__(self):
        # Store request timestamps for each IP/user
        self.requests = defaultdict(deque)

    def is_allowed(self, key, max_requests=5, window_seconds=300):
        """
        Check if a request is allowed based on rate limiting rules

        Args:
            key: Identifier (IP address, user ID, etc.)
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds

        Returns:
            bool: True if request is allowed, False if rate limited
        """
        now = time.time()
        window_start = now - window_seconds

        # Clean old requests outside the window
        request_times = self.requests[key]
        while request_times and request_times[0] < window_start:
            request_times.popleft()

        # Check if we're under the limit
        if len(request_times) < max_requests:
            request_times.append(now)
            return True

        return False

    def get_reset_time(self, key, window_seconds=300):
        """Get the time when rate limit will reset for a key"""
        request_times = self.requests.get(key, deque())
        if not request_times:
            return 0

        # Time when the oldest request in the window will expire
        return request_times[0] + window_seconds


# Global rate limiter instance
rate_limiter = RateLimiter()


def rate_limit(max_requests=10, window_seconds=100, per_user=True):
    """
    Decorator to add rate limiting to Flask routes

    Args:
        max_requests: Maximum requests allowed in window
        window_seconds: Time window in seconds
        per_user: If True, limit per authenticated user; if False, per IP

    Returns:
        Decorated function
    """

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Determine the key for rate limiting
            if per_user and current_user.is_authenticated:
                key = f"user:{current_user.id}"
            else:
                # Use IP address, with X-Forwarded-For header support
                key = f"ip:{request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)}"

            # Check rate limit
            if not rate_limiter.is_allowed(key, max_requests, window_seconds):
                reset_time = rate_limiter.get_reset_time(key, window_seconds)
                retry_after = max(0, int(reset_time - time.time()))

                current_app.logger.warning(
                    f"Rate limit exceeded for {key} on route {request.endpoint}"
                )

                return (
                    jsonify(
                        {
                            "success": False,
                            "message": "Rate limit exceeded. Please try again later.",
                            "retry_after": retry_after,
                        }
                    ),
                    429,
                )

            return f(*args, **kwargs)

        return decorated_function

    return decorator


def rate_limit_auth(max_requests=5, window_seconds=900):
    """
    Specialized rate limiter for authentication endpoints
    More restrictive by default (5 attempts per 15 minutes)
    """
    return rate_limit(
        max_requests=max_requests, window_seconds=window_seconds, per_user=False
    )


def rate_limit_api(max_requests=60, window_seconds=60):
    """
    Rate limiter for API endpoints
    Default: 60 requests per minute per user
    """
    return rate_limit(
        max_requests=max_requests, window_seconds=window_seconds, per_user=True
    )


def rate_limit_admin(max_requests=10, window_seconds=60):
    """
    Rate limiter for admin endpoints
    More restrictive: 10 requests per minute per user
    """
    return rate_limit(
        max_requests=max_requests, window_seconds=window_seconds, per_user=True
    )
