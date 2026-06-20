---
title: About
eyebrow: Profile
lead: A compact overview of my background, technical lens, and the kinds of problems I like to solve.
permalink: /about/
---

<section class="content-grid">
  <div>
    <h2>Research-driven engineering</h2>
    <p>
      I work on AI systems that need more than a good prompt. My interests are in
      building structured workflows around language models, especially when the
      task requires memory, tool use, retrieval, or domain-specific constraints.
    </p>
    <p>
      My transition from theoretical physics to AI engineering shaped how I think:
      start from first principles, make the system observable, and evaluate it
      with enough rigor that we can trust the outputs in real settings.
    </p>
  </div>

  <aside class="card">
    <p class="eyebrow">At a glance</p>
    <div class="stack-list">
      <div>
        <h3>Location</h3>
        <p>{{ site.data.profile.location }}</p>
      </div>
      <div>
        <h3>Current focus</h3>
        <p>{{ site.data.profile.focus.description }}</p>
      </div>
    </div>
  </aside>
</section>

<section class="panel">
  <div class="section-heading">
    <div>
      <p class="eyebrow">Core Themes</p>
      <h2>What I like working on</h2>
    </div>
  </div>
  <div class="card-grid">
    {% for theme in site.data.profile.research_themes %}
      <article class="card">
        <h3>{{ theme.title }}</h3>
        <p>{{ theme.description }}</p>
      </article>
    {% endfor %}
  </div>
</section>

<section class="panel">
  <div class="section-heading">
    <div>
      <p class="eyebrow">Education</p>
      <h2>Academic foundation</h2>
    </div>
  </div>
  <div class="card-grid compact-grid">
    {% for item in site.data.profile.education %}
      <article class="card">
        <h3>{{ item.degree }}</h3>
        <p>{{ item.school }}</p>
      </article>
    {% endfor %}
  </div>
</section>
