import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

class OHLCVLoader:
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL", "postgresql://localhost/crypto_data")
        self.engine = create_engine(self.db_url)

    def fetch_ohlcv(
        self,
        pair: str,
        start_time: str = None,
        end_time: str = None
    ) -> pd.DataFrame:

        query = f"""
            SELECT
                time,
                open,
                high,
                low,
                close,
                volume,
                num_trades
            FROM ohlcv_data
            WHERE pair = '{pair}'
        """

        if start_time:
            query += f" AND time >= '{start_time}'"
        if end_time:
            query += f" AND time < '{end_time}'"

        query += " ORDER BY time ASC"

        df = pd.read_sql(query, self.engine)
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)

        return df
