from datetime import datetime, timezone

now_utc = datetime.now(timezone.utc)
now_local = datetime.now()

print(f"UTC:   {now_utc.isoformat()}")
print(f"Local: {now_local.isoformat()}")
