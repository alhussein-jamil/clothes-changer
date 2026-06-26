#!/usr/bin/env python3
"""Add a user to the database."""

import argparse
import logging
import sys

from clothes_changer.db.database import Database, DatabaseError
from clothes_changer.utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Add a Clothes Changer user")
    parser.add_argument("username")
    parser.add_argument("password")
    parser.add_argument("--credits", type=int, default=10)
    parser.add_argument("--admin", action="store_true")
    args = parser.parse_args()

    logger.info(
        "Creating user %r (credits=%d, admin=%s)",
        args.username,
        args.credits,
        args.admin,
    )
    db = Database()
    try:
        db.register_user(args.username, args.password, credits=args.credits, is_admin=args.admin)
        print(f"User '{args.username}' created")
    except DatabaseError as e:
        logger.error("Failed to create user: %s", e)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
