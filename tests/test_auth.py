import pytest

from clothes_changer.db.database import Database, DatabaseError


def test_register_and_authenticate(db: Database):
    db.register_user("alice", "password123", credits=5)
    assert db.authenticate("alice", "password123")
    assert not db.authenticate("alice", "wrongpass")
    user = db.get_user("alice")
    assert user is not None
    assert user.credits == 5


def test_duplicate_user(db: Database):
    db.register_user("bob", "password123")
    with pytest.raises(DatabaseError):
        db.register_user("bob", "password456")


def test_deduct_credit(db: Database):
    db.register_user("carol", "password123", credits=2)
    assert db.deduct_credit("carol")
    assert db.get_credits("carol") == 1
    assert db.deduct_credit("carol")
    assert not db.deduct_credit("carol")
