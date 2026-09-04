"""
Check connectivity to an Azure Blob Storage account.

Builds a connection string from the four standard components and verifies
access by listing the containers in the account.

Configuration is read from environment variables (preferred), with an
optional fallback to a local `.env` file if python-dotenv is installed:

    AZURE_DEFAULT_ENDPOINTS_PROTOCOL   (default: "https")
    AZURE_ACCOUNT_NAME
    AZURE_ACCOUNT_KEY
    AZURE_ENDPOINT_SUFFIX              (default: "core.windows.net")

Usage:
    pip install azure-storage-blob
    python check_azure_blob.py
"""

import os
import sys

try:
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import AzureError
except ImportError:
    sys.exit(
        "Missing dependency. Install it with:\n"
        "    pip install azure-storage-blob"
    )

# Optional: load a local .env file if python-dotenv is available.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def build_connection_string() -> str:
    protocol = os.getenv("AZURE_DEFAULT_ENDPOINTS_PROTOCOL", "https")
    account_name = os.getenv("AZURE_ACCOUNT_NAME")
    account_key = os.getenv("AZURE_ACCOUNT_KEY")
    endpoint_suffix = os.getenv("AZURE_ENDPOINT_SUFFIX", "core.windows.net")

    missing = [
        name
        for name, value in (
            ("AZURE_ACCOUNT_NAME", account_name),
            ("AZURE_ACCOUNT_KEY", account_key),
        )
        if not value
    ]
    if missing:
        sys.exit("Missing required environment variables: " + ", ".join(missing))

    return (
        f"DefaultEndpointsProtocol={protocol};"
        f"AccountName={account_name};"
        f"AccountKey={account_key};"
        f"EndpointSuffix={endpoint_suffix}"
    )


def check_connection(connection_string: str) -> int:
    try:
        client = BlobServiceClient.from_connection_string(connection_string)
        # A lightweight authenticated call that forces a round-trip to Azure.
        account_info = client.get_account_information()
        containers = list(client.list_containers())
    except AzureError as exc:
        print(f"[FAIL] Could not connect to Azure Blob Storage: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - surface anything unexpected
        print(f"[FAIL] Unexpected error: {exc}")
        return 1

    print("[OK] Connected to Azure Blob Storage")
    print(f"      Account URL : {client.url}")
    print(f"      Account kind: {account_info.get('account_kind')}")
    print(f"      SKU         : {account_info.get('sku_name')}")
    print(f"      Containers  : {len(containers)}")
    for container in containers:
        print(f"        - {container['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(check_connection(build_connection_string()))
