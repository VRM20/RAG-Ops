"""Document-level access control.

Every chunk carries the ACL groups of its source document in its payload. Retrieval
always runs with a server-side filter `acl_groups ANY user.groups`, so chunks a user
may not read are never scored, never reranked and never reach the LLM prompt. This is
"security trimming" at the retrieval layer: prompt-level instructions are not a
security boundary, the index filter is.
"""

from __future__ import annotations

from ragops.config import get_users
from ragops.types import UserContext

PUBLIC_GROUP = "public"


class UnknownUserError(LookupError):
    pass


def resolve_user(user_id: str | None) -> UserContext:
    """Resolve a user id to its groups.

    Production: derive groups from verified OIDC/JWT claims instead of a lookup table.
    """
    if not user_id:
        return UserContext(user_id="anonymous", name="Anonymous", groups=[PUBLIC_GROUP])
    profile = get_users().get(user_id)
    if profile is None:
        raise UnknownUserError(user_id)
    groups = sorted({PUBLIC_GROUP, *profile.groups})
    return UserContext(user_id=profile.user_id, name=profile.name, groups=groups)


def can_read(user: UserContext, acl_groups: list[str]) -> bool:
    return bool(set(user.groups) & set(acl_groups))


def all_known_groups() -> list[str]:
    groups = {PUBLIC_GROUP}
    for profile in get_users().values():
        groups.update(profile.groups)
    return sorted(groups)
