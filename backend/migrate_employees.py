"""
One-off migration: fixes the `employees` table schema in attendance.db
without losing existing rows.

Run this ONCE, from the same folder as attendance.db, before starting app.py:

    python migrate_employees.py

What it does:
  1. Prints the current (broken) schema so you can see what's actually there.
  2. Renames the old table to employees_old_backup (nothing is deleted).
  3. Creates a fresh `employees` table with the schema app.py expects.
  4. Copies over any old columns that match by name (case-insensitive),
     filling in a person_id if the old table used a different column name
     for it (e.g. 'id', 'emp_id', 'employee_id').
  5. Leaves employees_old_backup in place as a safety net - you can drop it
     manually later once you've confirmed the data looks right.
"""
import sqlite3
import sys
from datetime import datetime

DB_PATH = "attendance.db"

# Common alternate names the old ID column might have had.
ID_COLUMN_CANDIDATES = ["person_id", "id", "emp_id", "employee_id", "employeeid", "empid"]

NEW_COLUMNS = ["person_id", "name", "department", "designation", "phone", "email", "created_at", "updated_at"]


def get_columns(conn, table):
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [r[1] for r in rows]  # r[1] is the column name


def main():
    conn = sqlite3.connect(DB_PATH)

    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]

    if "employees" not in tables:
        print("No 'employees' table found - nothing to migrate. "
              "Just run app.py normally; it will create the table fresh.")
        return

    old_columns = get_columns(conn, "employees")
    print(f"Current 'employees' columns: {old_columns}")

    if old_columns == NEW_COLUMNS or set(NEW_COLUMNS).issubset(set(old_columns)):
        print("Schema already looks correct. No migration needed.")
        return

    row_count = conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0]
    print(f"Found {row_count} existing row(s) to preserve.")

    # Figure out which old column holds the person's ID.
    id_col = next((c for c in ID_COLUMN_CANDIDATES if c in old_columns), None)
    if id_col is None:
        print(f"WARNING: could not find an ID-like column among {old_columns}. "
              f"Backing up the old table but starting employees empty - "
              f"you'll need to re-enter names manually via /enroll or tell me "
              f"the old column name so I can fix this script.")
        id_col = None

    backup_name = f"employees_old_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    conn.execute(f"ALTER TABLE employees RENAME TO {backup_name}")
    print(f"Renamed old table to '{backup_name}' (kept, not deleted).")

    conn.execute(
        """
        CREATE TABLE employees (
            person_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT,
            designation TEXT,
            phone TEXT,
            email TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    print("Created new 'employees' table with the correct schema.")

    if id_col is not None:
        now_iso = datetime.now().isoformat(timespec="seconds")
        old_rows = conn.execute(f"SELECT * FROM {backup_name}").fetchall()
        old_col_index = {c: i for i, c in enumerate(old_columns)}

        def get_val(row, col, default=""):
            i = old_col_index.get(col)
            return row[i] if i is not None and row[i] is not None else default

        migrated = 0
        for row in old_rows:
            person_id = str(get_val(row, id_col)).strip()
            if not person_id:
                continue
            name = get_val(row, "name") or person_id
            department = get_val(row, "department")
            designation = get_val(row, "designation")
            phone = get_val(row, "phone")
            email = get_val(row, "email")
            created_at = get_val(row, "created_at") or now_iso
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO employees
                    (person_id, name, department, designation, phone, email, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (person_id, name, department, designation, phone, email, created_at, now_iso),
                )
                migrated += 1
            except sqlite3.Error as e:
                print(f"  Skipped row for '{person_id}': {e}")

        conn.commit()
        print(f"Migrated {migrated}/{len(old_rows)} row(s) into the new table.")
    else:
        conn.commit()

    print("\nDone. You can now run: python app.py")
    print(f"(Old data is safely kept in '{backup_name}' if you need to double check anything.)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("Migration failed - your original data has NOT been touched "
              "(sqlite only commits on success unless stated above).")
        raise


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--db":
    DB_PATH = sys.argv[2]
