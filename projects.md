---
title: Projects
eyebrow: Project Portfolio
lead: Selected systems and open-source work across agentic AI, retrieval, and scientific tooling.
permalink: /projects/
---

<section class="panel">
  <div class="card-grid">
    {% for project in site.data.profile.projects %}
      <article class="card project-card">
        <p class="meta">{{ project.label }}</p>
        <h2>{{ project.title }}</h2>
        <p>{{ project.description }}</p>
        <div class="tag-list">
          {% for item in project.stack %}
            <span>{{ item }}</span>
          {% endfor %}
        </div>
        <a class="text-link" href="{{ project.url }}">{{ project.cta }}</a>
      </article>
    {% endfor %}
  </div>
</section>
