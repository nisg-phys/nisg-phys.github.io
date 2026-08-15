---
title: Research
permalink: /research/
classes: wide
---

{{ site.data.profile.research_intro }}

## Current interests

<article class="content-card">
  <ul class="content-list">
  {% for interest in site.data.profile.research_interests %}
    <li>{{ interest }}</li>
  {% endfor %}
  </ul>
</article>

## Physics background

{{ site.data.profile.physics_research_summary }}

## Academic links

<div class="list-grid">
{% for link in site.data.profile.academic_links %}
  <article class="list-card">
    <h3>{{ link.label }}</h3>
    <p><a href="{{ link.url }}">{{ link.url }}</a></p>
  </article>
{% endfor %}
</div>

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

## Upcoming preprint

<div class="list-grid">
{% for item in site.data.profile.upcoming_preprints %}
  <article class="list-card">
    <h3>{{ item.title }}</h3>
  </article>
{% endfor %}
</div>

## Selected talks

<div class="list-grid">
{% for talk in site.data.profile.selected_talks %}
  <article class="list-card">
    <p class="section-kicker">{{ talk.period }}</p>
    <h3>{{ talk.title }}</h3>
    <p>{{ talk.venue }}</p>
  </article>
{% endfor %}
</div>
