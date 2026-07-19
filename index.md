---
title: "Nishant Gupta"
layout: splash
permalink: /
author_profile: false
excerpt: "AI researcher and engineer focused on agentic systems, evaluation pipelines, and scientific machine intelligence."
header:
  overlay_color: "#111827"
  overlay_filter: 0.35
classes: wide
---

<div class="home-intro">
  <p class="home-intro__lead">{{ site.data.profile.summary }}</p>
  <div class="home-intro__actions">
    <a href="/projects/" class="btn btn--primary">View Projects</a>
    <a href="/contact/" class="btn btn--inverse">Get in Touch</a>
  </div>
</div>

<div class="detail-grid">
  <article class="detail-card">
    <p class="section-kicker">Current focus</p>
    <h2>{{ site.data.profile.focus.title }}</h2>
    <p>{{ site.data.profile.focus.description }}</p>
  </article>
  <article class="detail-card">
    <p class="section-kicker">Location</p>
    <h2>{{ site.data.profile.location }}</h2>
    <p>Open to work across AI engineering, evaluation, and research-focused systems.</p>
  </article>
</div>

## Research areas

<div class="card-grid">
{% for theme in site.data.profile.research_themes %}
  <article class="content-card">
    <h3>{{ theme.title }}</h3>
    <p>{{ theme.description }}</p>
  </article>
{% endfor %}
</div>

## Selected projects

<div class="card-grid">
{% for project in site.data.profile.projects %}
  <article class="content-card">
    <p class="section-kicker">{{ project.label }}</p>
    <h3>
      {% if project.url %}
      <a href="{{ project.url }}">{{ project.title }}</a>
      {% elsif project.links %}
      <a href="{{ project.links[0].url }}">{{ project.title }}</a>
      {% else %}
      {{ project.title }}
      {% endif %}
    </h3>
    <p>{{ project.description }}</p>
    <p class="meta-line">Stack: {{ project.stack | join: " • " }}</p>
    {% if project.links %}
    <p class="project-links">
      {% for link in project.links %}
      <a href="{{ link.url }}">{{ link.label }}</a>{% unless forloop.last %} <span aria-hidden="true">•</span> {% endunless %}
      {% endfor %}
    </p>
    {% endif %}
  </article>
{% endfor %}
</div>
