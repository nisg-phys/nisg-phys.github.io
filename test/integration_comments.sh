#!/usr/bin/env bash
set -euo pipefail

tmp_dir="$(mktemp -d)"
tmp_override="${tmp_dir}/comments-test-override.yml"
tmp_site="${tmp_dir}/site"

cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

cat >"${tmp_override}" <<'YAML'
defaults:
  - scope:
      path: "_pages"
      type: pages
    values:
      giscus_comments: true

disqus_shortname: false

giscus:
  repo: alshedivat/al-folio
  repo_id: R_kgDOExample
  category: Comments
  category_id: DIC_kwDOExample
YAML

bundle exec jekyll build --config "_config.yml,${tmp_override}" -d "${tmp_site}" >/dev/null

about_page="${tmp_site}/about/index.html"

grep -q 'https://giscus.app/client.js' "${about_page}"
grep -q 'id="giscus_thread"' "${about_page}"
if grep -q 'giscus comments misconfigured' "${about_page}"; then
  echo "unexpected giscus misconfiguration warning in ${about_page}" >&2
  exit 1
fi

echo "comments integration checks passed"
