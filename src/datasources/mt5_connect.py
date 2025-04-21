import MetaTrader5 as mt5
import os


MT5_BROKER_SERVER="mt5-demo01.pepperstone.com"
MT5_BROKER_LOGIN=61342026
MT5_BROKER_PASSWORD="@cTrader3DEMO"


def connect():
    """
    Connects to the MetaTrader 5 terminal using credentials from .env file.

    Requires the following environment variables:
    YES!! You do have to create a .env file for your own with credentials.
    It is not in the repo. So create it at root level.
    The .env file should contain:
        MT5_BROKER_LOGIN=your_login
        MT5_BROKER_SERVER=your_server
        MT5_BROKER_PASSWORD=your_password

    Provides feedback on connection issues.

    4 digit function signature: 0417
    """
    try:
        # login_raw = os.getenv('MT5_BROKER_LOGIN')
        # server = os.getenv('MT5_BROKER_SERVER', "").strip()
        # password = os.getenv('MT5_BROKER_PASSWORD', "").strip()
        login_raw = MT5_BROKER_LOGIN
        server = MT5_BROKER_SERVER
        password = MT5_BROKER_PASSWORD

        # Validate login
        if not login_raw:
            return False
        try:
            # login = int(login_raw.strip())
            login = int(login_raw)
        except ValueError:
            return False

        # Validate the remaining credentials
        if not server:
            return False
        if not password:
            return False

        # Initialize MT5 terminal
        if not mt5.initialize(login=login, server=server, password=password):
            error_code, error_msg = mt5.last_error()
            handle_connection_error(error_code, server, login)
            return False

        # Successful initialization log

        account_info = mt5.account_info()
        if account_info:
            print(
                f"Connected to MetaTrader 5. Account: {account_info.login}"
            )
        else:
            print(
                "Failed to retrieve account information. Check your connection."
            )

        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def disconnect():
    """
    Closes the MT5 connection safely.

    4 digit function signature: 0429
    """
    mt5.shutdown()


def handle_connection_error(error_code, server, login):
    """
    Provides feedback on connection issues.

    4 digit function signature: 0431
    """
    if error_code == -6:
        print(
            "Invalid login or password. Check your credentials."
        )
    elif error_code == 5:
        print(
            "Invalid server. Check your server address."
        )
    elif error_code == 10014:
        print(
            "Connection timeout. Check your internet connection."
        )
    else:
        print(
            f"Connection error {error_code}. Check your credentials and server."
        )


# Run connection test
if __name__ == "__main__":
    if connect():
        print(mt5.terminal_info())  # Display connection info
        disconnect()


