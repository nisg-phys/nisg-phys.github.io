---
title: Research
permalink: /research/
classes: wide
---

## Theoretical physics publications

Publication records and citation details are available through my [INSPIRE-HEP profile]({{ site.data.profile.theoretical_physics_publications.citation_summary_url }}).

<div class="publication-list">
{% for paper in site.data.profile.theoretical_physics_publications.items %}
  <article class="publication-item">
    <h3>{{ paper.title }}</h3>
    <p>{{ paper.authors | join: ", " }}</p>
    <p><a href="{{ paper.url }}">{{ paper.venue }}</a></p>
  </article>
{% endfor %}
</div>

## AI research interests

<div class="card-grid">
{% for theme in site.data.profile.research_themes %}
  <article class="content-card">
    <h3>{{ theme.title }}</h3>
    <p>{{ theme.description }}</p>
  </article>
{% endfor %}
</div>

## Technical toolkit

<div class="list-grid">
{% for group in site.data.profile.skill_groups %}
  <article class="list-card">
    <h3>{{ group.title }}</h3>
    <p>{{ group.items | join: ", " }}</p>
  </article>
{% endfor %}
</div>
