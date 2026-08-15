---
layout: splash
permalink: /
author_profile: false
header:
  overlay_color: "#111827"
  overlay_filter: 0.35
classes: wide
---

<div class="home-intro">
  {% for paragraph in site.data.profile.home_intro %}
  <p class="home-intro__lead">{{ paragraph }}</p>
  {% endfor %}
</div>

<article class="content-card">
  <p class="section-kicker">Now</p>
  <h2>{{ site.data.profile.home_now.title }}</h2>
  <p>{{ site.data.profile.home_now.description }}</p>
</article>

<div class="card-grid">
{% for item in site.data.profile.home_links %}
  <article class="content-card">
    <p class="section-kicker">{{ item.label }}</p>
    <h3><a href="{{ item.url }}">{{ item.title }}</a></h3>
    <p>{{ item.description }}</p>
  </article>
{% endfor %}
</div>
