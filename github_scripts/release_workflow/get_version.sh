#!/usr/bin/env bash
set -euo pipefail

if [[ "${GITHUB_REF_TYPE}" != "tag" ]]; then
    echo "Workflow expected to run on tag push but received: ${GITHUB_REF_TYPE}" >&2
    exit 1
fi

tag="${GITHUB_REF_NAME}"
prefix="release/"

if [[ "${tag}" == ${prefix}* ]]; then
    version="${tag#${prefix}}"
else
    version="${tag}"
fi

if [[ -z "${version}" ]]; then
    echo "Unable to derive release version from tag name." >&2
    exit 1
fi

release_branch=""
git fetch --quiet origin "+refs/heads/*:refs/remotes/origin/*"
release_branch=$(git for-each-ref --format='%(refname:strip=3)' --contains "${GITHUB_SHA}" "refs/remotes/origin/release/*" | head -n 1 || true)

if [[ -z "${release_branch}" ]]; then
    release_branch=$(git for-each-ref --format='%(refname:strip=3)' --contains "${GITHUB_SHA}" "refs/remotes/origin" | head -n 1 || true)
fi

if [[ -z "${release_branch}" ]]; then
    if [[ -n "${GITHUB_EVENT_PATH:-}" && -s "${GITHUB_EVENT_PATH}" ]]; then
        default_branch=$(jq -r '.repository.default_branch // empty' "${GITHUB_EVENT_PATH}" 2>/dev/null || echo "")
        if [[ -n "${default_branch}" ]]; then
            release_branch="${default_branch}"
        fi
    fi
fi

if [[ -z "${release_branch}" ]]; then
    release_branch="main"
fi

echo "release-version=${version}" >> "${GITHUB_OUTPUT}"
echo "release-branch=${release_branch}" >> "${GITHUB_OUTPUT}"