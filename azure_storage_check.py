"""Report per-container blob usage for the configured Azure Blob account."""

import os
import sys
from collections import Counter

from azure.storage.blob import BlobServiceClient

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} PiB"


def main() -> int:
    conn = (
        f"DefaultEndpointsProtocol={os.getenv('AZURE_DEFAULT_ENDPOINTS_PROTOCOL', 'https')};"
        f"AccountName={os.environ['AZURE_ACCOUNT_NAME']};"
        f"AccountKey={os.environ['AZURE_ACCOUNT_KEY']};"
        f"EndpointSuffix={os.getenv('AZURE_ENDPOINT_SUFFIX', 'core.windows.net')}"
    )
    client = BlobServiceClient.from_connection_string(conn)
    info = client.get_account_information()

    print(f"Account : {client.url}")
    print(f"Kind    : {info.get('account_kind')}   SKU: {info.get('sku_name')}")
    print()

    grand_bytes = 0
    grand_blobs = 0
    for container in client.list_containers():
        cc = client.get_container_client(container["name"])
        total = 0
        count = 0
        tiers = Counter()
        for blob in cc.list_blobs():
            total += blob.size or 0
            count += 1
            tiers[blob.blob_tier or "unknown"] += 1
        grand_bytes += total
        grand_blobs += count
        tier_str = ", ".join(f"{k}={v}" for k, v in sorted(tiers.items())) or "-"
        print(f"  {container['name']}")
        print(f"    blobs : {count:,}")
        print(f"    size  : {human(total)}")
        print(f"    tiers : {tier_str}")
        print()

    print(f"TOTAL   : {human(grand_bytes)} across {grand_blobs:,} blobs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
