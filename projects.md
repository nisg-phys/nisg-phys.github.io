---
title: Projects
permalink: /projects/
classes: wide
---

Selected work in multi-agent systems, retrieval, and scientific AI tooling.

<div class="card-grid">
{% for project in site.data.profile.projects %}
  <article class="content-card">
    <p class="section-kicker">{{ project.label }}</p>
    <h2>{% if project.url %}<a href="{{ project.url }}">{{ project.title }}</a>{% else %}{{ project.title }}{% endif %}</h2>
    <p>{{ project.description }}</p>
    <p class="meta-line">Stack: {{ project.stack | join: " • " }}</p>
    {% if project.links %}
    <p class="project-links">
      {% for link in project.links %}
      <a href="{{ link.url }}" class="btn btn--primary btn--small">{{ link.label }}</a>
      {% endfor %}
    </p>
    {% elsif project.url and project.cta %}
    <p><a href="{{ project.url }}" class="btn btn--primary btn--small">{{ project.cta }}</a></p>
    {% endif %}
  </article>
{% endfor %}
</div>
