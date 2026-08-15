---
title: About
permalink: /about/
classes: wide
---

{% for paragraph in site.data.profile.about_story %}
{{ paragraph }}

{% endfor %}
