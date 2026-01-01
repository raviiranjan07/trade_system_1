"""
Custom exceptions for the Trading System.

Provides specific, user-friendly error types for different failure scenarios.
"""


class TradingSystemError(Exception):
    """Base exception for all trading system errors."""

    def __init__(self, message: str, suggestion: str = None):
        self.message = message
        self.suggestion = suggestion
        super().__init__(self.format_message())

    def format_message(self) -> str:
        msg = self.message
        if self.suggestion:
            msg += f"\n\nSuggestion: {self.suggestion}"
        return msg


class DatabaseConnectionError(TradingSystemError):
    """Raised when database connection fails."""

    def __init__(self, db_url: str, original_error: Exception = None):
        # Parse connection info from URL (hide password)
        self.db_url = self._mask_password(db_url)
        self.original_error = original_error

        message = f"""Unable to connect to database

Connection: {self.db_url}

Possible causes:
  1. PostgreSQL service is not running
  2. Wrong host/port in DATABASE_URL
  3. Database does not exist
  4. Invalid credentials"""

        if original_error:
            # Extract key info from original error
            error_str = str(original_error)
            if "Connection refused" in error_str:
                message += "\n  5. Firewall blocking the connection"

        suggestion = """Check the following:
  - Run: pg_isready -h localhost -p 5432
  - Verify DATABASE_URL in .env or config/config.yaml
  - Ensure PostgreSQL service is running"""

        super().__init__(message, suggestion)

    @staticmethod
    def _mask_password(url: str) -> str:
        """Hide password in connection URL for display."""
        import re
        return re.sub(r'://([^:]+):([^@]+)@', r'://\1:****@', url)


class ConfigurationError(TradingSystemError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, field: str, issue: str, current_value=None):
        self.field = field
        self.issue = issue
        self.current_value = current_value

        message = f"Configuration error in '{field}': {issue}"
        if current_value is not None:
            message += f"\n  Current value: {current_value}"

        suggestion = "Check config/config.yaml and ensure all required fields are set correctly."

        super().__init__(message, suggestion)


class DataValidationError(TradingSystemError):
    """Raised when data fails validation checks."""

    def __init__(self, message: str, data_source: str = None):
        self.data_source = data_source

        full_message = f"Data validation failed: {message}"
        if data_source:
            full_message += f"\n  Source: {data_source}"

        suggestion = "Ensure your data source contains valid, complete OHLCV data."

        super().__init__(full_message, suggestion)


class MissingDataError(TradingSystemError):
    """Raised when required data files are not found."""

    def __init__(self, file_path: str, data_type: str = None):
        self.file_path = file_path
        self.data_type = data_type

        message = f"Required data file not found: {file_path}"

        if data_type:
            suggestion = f"Run the {data_type} pipeline stage first to generate this file."
        else:
            suggestion = "Ensure previous pipeline stages have been run successfully."

        super().__init__(message, suggestion)


class PipelineStageError(TradingSystemError):
    """Raised when a pipeline stage fails."""

    def __init__(self, stage: str, original_error: Exception = None):
        self.stage = stage
        self.original_error = original_error

        message = f"Pipeline stage '{stage}' failed"
        if original_error:
            message += f": {str(original_error)}"

        suggestion = f"Check the logs for details. You may need to fix the issue and re-run the '{stage}' stage."

        super().__init__(message, suggestion)
