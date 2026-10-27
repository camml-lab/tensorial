#!/bin/bash
# Release a new version: bump the version on the current branch (usually develop), merge it into
# main and push both.  CI then publishes to PyPI and creates the v<version> tag and GitHub release.
#
# Usage: ./release.sh <version>

set -e

PACKAGE="tensorial"
# The publish job only runs on the main repository (camml-lab/tensorial), not on forks
REMOTE="camml"
RELEASE_BRANCH="main"
VERSION_FILE="src/${PACKAGE}/__init__.py"

version=$1
if [[ ! $version =~ ^[0-9]+\.[0-9]+\.[0-9]+([a-z0-9.]*)$ ]]; then
    echo "Usage: $0 <version> (e.g. 0.6.6)"
    exit 1
fi

if [[ -n $(git status --porcelain --untracked-files=no) ]]; then
    echo "There are uncommitted changes, please commit or stash them first"
    exit 1
fi

current_version=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' $VERSION_FILE)
if [[ $version == "$current_version" ]]; then
    echo "The version is already ${version}"
    exit 1
fi

current_branch=$(git rev-parse --abbrev-ref HEAD)

# Make sure we have everything from the remote, so that the push at the end doesn't get rejected
git fetch $REMOTE
if git show-ref --verify --quiet refs/remotes/$REMOTE/$current_branch \
    && ! git merge-base --is-ancestor $REMOTE/$current_branch $current_branch; then
    echo "${REMOTE}/${current_branch} has commits that ${current_branch} doesn't, please merge them first"
    exit 1
fi

while true; do
    read -p "Release version ${version} (currently ${current_version}) from ${current_branch} to ${REMOTE}? " yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

set -x

sed -i "s/^__version__ = .*/__version__ = \"${version}\"/" $VERSION_FILE
git add $VERSION_FILE
git commit -m "chore(release): ${version}"

# Merge into the release branch (creating it if needed)
if git show-ref --verify --quiet refs/heads/$RELEASE_BRANCH; then
    git checkout $RELEASE_BRANCH
    # Bring in any releases made from elsewhere
    if git show-ref --verify --quiet refs/remotes/$REMOTE/$RELEASE_BRANCH; then
        git merge --ff-only $REMOTE/$RELEASE_BRANCH
    fi
elif git show-ref --verify --quiet refs/remotes/$REMOTE/$RELEASE_BRANCH; then
    git checkout -b $RELEASE_BRANCH --track $REMOTE/$RELEASE_BRANCH
else
    git checkout -b $RELEASE_BRANCH
fi
git merge --no-edit $current_branch
git checkout $current_branch

# Pushing the release branch triggers the publish job
git push $REMOTE $RELEASE_BRANCH $current_branch
