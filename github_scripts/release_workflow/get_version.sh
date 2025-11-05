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

echo "release-version=${version}" >> "${GITHUB_OUTPUT}"
echo "RELEASE_VERSION=${version}" >> "${GITHUB_ENV}"