"""
OHLCV Data Loader

Fetches OHLCV (Open, High, Low, Close, Volume) data from PostgreSQL/TimescaleDB.
"""

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, ProgrammingError

from exceptions import DatabaseConnectionError, DataValidationError

load_dotenv()


class OHLCVLoader:
    """
    Loads OHLCV data from a PostgreSQL database.

    Usage:
        loader = OHLCVLoader()
        df = loader.fetch_ohlcv("BTCUSDT", start_time="2023-01-01")
    """

    def __init__(self, db_url: str = None):
        """
        Initialize the loader with database connection.

        Args:
            db_url: PostgreSQL connection URL. Falls back to DATABASE_URL env var.

        Raises:
            DatabaseConnectionError: If connection to database fails.
        """
        self.db_url = db_url or os.getenv("DATABASE_URL", "postgresql://localhost/crypto_data")
        self.engine = self._create_engine()

    def _create_engine(self):
        """Create database engine with connection validation."""
        try:
            engine = create_engine(self.db_url)
            # Test the connection immediately
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return engine
        except OperationalError as e:
            raise DatabaseConnectionError(self.db_url, original_error=e)
        except Exception as e:
            raise DatabaseConnectionError(self.db_url, original_error=e)

    def test_connection(self) -> bool:
        """
        Test if the database connection is working.

        Returns:
            True if connection is successful.

        Raises:
            DatabaseConnectionError: If connection fails.
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            raise DatabaseConnectionError(self.db_url, original_error=e)

    def fetch_ohlcv(
        self,
        pair: str,
        start_time: str = None,
        end_time: str = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a trading pair.

        Args:
            pair: Trading pair symbol (e.g., "BTCUSDT")
            start_time: Start date in ISO format (YYYY-MM-DD)
            end_time: End date in ISO format (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data indexed by time.

        Raises:
            DatabaseConnectionError: If database query fails.
            DataValidationError: If no data is returned.
        """
        # Build query with parameterized values (prevents SQL injection)
        query = """
            SELECT
                time,
                open,
                high,
                low,
                close,
                volume,
                num_trades
            FROM ohlcv_data
            WHERE pair = :pair
        """

        params = {"pair": pair}

        if start_time:
            query += " AND time >= :start_time"
            params["start_time"] = start_time
        if end_time:
            query += " AND time < :end_time"
            params["end_time"] = end_time

        query += " ORDER BY time ASC"

        try:
            df = pd.read_sql(text(query), self.engine, params=params)
        except OperationalError as e:
            raise DatabaseConnectionError(self.db_url, original_error=e)
        except ProgrammingError as e:
            # Table doesn't exist or column issues
            error_msg = str(e)
            if "does not exist" in error_msg:
                raise DataValidationError(
                    f"Table 'ohlcv_data' not found in database",
                    data_source=self.db_url
                )
            raise DatabaseConnectionError(self.db_url, original_error=e)

        # Validate results
        if df.empty:
            raise DataValidationError(
                f"No OHLCV data found for {pair}" +
                (f" between {start_time} and {end_time}" if start_time or end_time else ""),
                data_source=self.db_url
            )

        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)

        return df
