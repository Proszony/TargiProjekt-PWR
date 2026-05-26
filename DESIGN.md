---
name: Fair Monitor
description: Multi-camera booth analytics with a focus-first control room interface.
colors:
  canvas-night: "#071018"
  shell-deep: "#0a131b"
  shell-surface: "#0d1721"
  panel-surface: "#132230"
  panel-hover: "#182b3b"
  panel-selected: "#1a3040"
  input-surface: "#101923"
  chip-surface: "#112130"
  border-strong: "#1f3141"
  border-accent: "#254255"
  border-soft: "#23394b"
  text-primary: "#f2f7fb"
  text-secondary: "#dce7f0"
  text-muted: "#9fb1c0"
  text-subtle: "#90a5b6"
  text-faint: "#6d8091"
  accent-aqua: "#75d3e0"
  accent-aqua-strong: "#7de3e1"
  accent-aqua-soft: "#98edeb"
  accent-blue: "#38bdf8"
  accent-green: "#22c55e"
  accent-amber: "#f59e0b"
  accent-violet: "#a78bfa"
  accent-rose: "#f43f5e"
  danger-surface: "#4d1820"
  danger-border: "#7a2b3a"
typography:
  display:
    fontFamily: "-apple-system, BlinkMacSystemFont, \"Segoe UI\", system-ui, sans-serif"
    fontSize: "30px"
    fontWeight: 700
    lineHeight: 1.15
  headline:
    fontFamily: "-apple-system, BlinkMacSystemFont, \"Segoe UI\", system-ui, sans-serif"
    fontSize: "24px"
    fontWeight: 700
    lineHeight: 1.2
  title:
    fontFamily: "-apple-system, BlinkMacSystemFont, \"Segoe UI\", system-ui, sans-serif"
    fontSize: "18px"
    fontWeight: 700
    lineHeight: 1.25
  body:
    fontFamily: "-apple-system, BlinkMacSystemFont, \"Segoe UI\", system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.45
  label:
    fontFamily: "-apple-system, BlinkMacSystemFont, \"Segoe UI\", system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 600
    lineHeight: 1.35
    letterSpacing: "0.02em"
rounded:
  sm: "8px"
  md: "10px"
  lg: "12px"
spacing:
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "18px"
  xxl: "22px"
components:
  button-primary:
    backgroundColor: "{colors.accent-aqua-strong}"
    textColor: "#06222b"
    rounded: "{rounded.sm}"
    padding: "10px 14px"
  button-secondary:
    backgroundColor: "{colors.panel-surface}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.sm}"
    padding: "10px 14px"
  panel-card:
    backgroundColor: "{colors.shell-surface}"
    textColor: "{colors.text-secondary}"
    rounded: "{rounded.lg}"
    padding: "18px"
  metric-card:
    backgroundColor: "{colors.shell-surface}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.lg}"
    padding: "16px 14px"
  input-field:
    backgroundColor: "{colors.input-surface}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.sm}"
    padding: "4px 10px"
  camera-selector:
    backgroundColor: "{colors.panel-surface}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.sm}"
    padding: "12px 14px"
---

# Design System: Fair Monitor

## Overview

**Creative North Star: "The Signal Atlas"**

Fair Monitor is a control-room product, not a dashboard spectacle. The interface is built around one idea: the operator should always know which camera is in focus, how that feed relates to the map, and whether runtime health is stable enough to trust the numbers. The design stays dark and restrained so live video, booth overlays, and telemetry can carry the visual emphasis instead of decorative chrome.

The system rejects equal-weight CCTV walls and generic SaaS dashboard habits. Panels are dense but calm, actions are grouped by operator task, and the map behaves like a spatial companion to the selected feed instead of a second competing destination. Saturated color is reserved for active selection, key actions, and analytics signals that matter right now.

Key Characteristics:
- Focus-first camera workflow, one primary feed with fast switching.
- Tonal dark surfaces with a cool aqua accent for current state and primary actions.
- Calm telemetry, dense enough for operators, never ornamental.
- Map and feed treated as a paired workspace.
- Editing tools remain available without disrupting live monitoring context.

## Colors

The palette is a cool, maritime-dark system: dense blue-black shells, steel-blue borders, and a restrained aqua signal color that only steps forward when the interface needs to say "this is live" or "this is selected."

### Primary
- **Signal Aqua** (`#75d3e0`): The clearest selection and live-state accent. Used for active map toggles, focus treatments, and guidance text that needs to feel technical rather than promotional.
- **Command Aqua** (`#7de3e1`): Reserved for primary actions such as starting runtime, accepting calibration coverage, and key export affordances.

### Secondary
- **Telemetry Blue** (`#38bdf8`): Used inside analytics charts and visit-oriented data marks where the interface needs a chart-specific highlight without borrowing the primary action color.
- **Occupancy Green** (`#22c55e`): Reserved for occupancy-positive metrics and active booth counts.

### Tertiary
- **Time Amber** (`#f59e0b`): Used for overlap emphasis, time-spent signals, and warning-adjacent attention that should remain calm.
- **Load Violet** (`#a78bfa`): Used sparingly for peak metrics and supplemental categorical contrast.
- **Alert Rose** (`#f43f5e`): Used for high-salience analytics counters and destructive pressure only when necessary.

### Neutral
- **Atlas Night** (`#071018`): The app canvas, used behind all major surfaces.
- **Shell Deep** (`#0a131b`): The darker of the panel layers, used for grouped toolbars and chart headers.
- **Shell Surface** (`#0d1721`): The default card and workspace shell.
- **Control Surface** (`#132230`): The interactive resting surface for standard buttons and selector items.
- **Selected Surface** (`#1a3040`): The active neutral state for toggles, selected selectors, and focused tabs.
- **Input Bed** (`#101923`): Used under fields, tables, and chart plot areas.
- **Steel Border** (`#1f3141`): The default stroke for panels, tables, and input frames.
- **Mist Text** (`#9fb1c0`): The default support text and metadata color.
- **Signal White** (`#f2f7fb`): Used for headings, primary values, and the highest-priority labels.

### Named Rules
**The Signal Reserve Rule.** Aqua is not decorative. If a surface uses Signal Aqua, it is either the current focus, the primary action, or a live-state cue.

**The Dark Room Rule.** Background layers may get lighter as they move upward in hierarchy, but they never flip to bright neutrals. Video and map overlays need the luminous headroom.

## Typography

**Display Font:** `-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif`
**Body Font:** `-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif`
**Label/Mono Font:** Same system sans stack, no separate mono token is currently used in the product.

**Character:** The typography is native, sharp, and operational. It should feel like a well-tuned desktop tool, not a branded campaign. Hierarchy comes from weight and spacing discipline rather than expressive type pairing.

### Hierarchy
- **Display** (700, 30px, 1.15): Used for the main shell heading and any rare screen-level anchors.
- **Headline** (700, 24px, 1.2): Used for major secondary-screen headers such as calibration or analytics overviews.
- **Title** (700, 18px, 1.25): Used for panel titles, dialog titles, and workspace section names.
- **Body** (400, 13px, 1.45): Used for helper copy, summaries, telemetry strings, and table-adjacent explanatory text.
- **Label** (600, 12px, 1.35, 0.02em letter spacing): Used for metric labels, tab labels, control hints, and compact status metadata.

### Named Rules
**The Native Confidence Rule.** Stay on the system sans stack. Product trust comes from clarity and platform familiarity, not from introducing a display font into controls.

## Elevation

This system is tonal rather than shadow-driven. Depth is conveyed by stacked dark surfaces, border contrast, and selected-state shifts instead of soft drop shadows. Panels sit on the canvas through color separation, not visual float. The effect should feel composed and technical, like a dim operator room with illuminated instrumentation.

### Named Rules
**The Flat-at-Rest Rule.** Surfaces do not lift with shadows in their idle state. State changes are shown through tint, border, and accent behavior first.

## Components

### Buttons
- **Character:** Buttons are compact control surfaces, never glossy badges.
- **Shape:** Compact soft rectangle, 8px radius.
- **Primary:** Command Aqua fill with dark text, used for runtime start, accept/apply actions, and key exports.
- **Hover / Focus:** Hover shifts the neutral or aqua surface lighter; focus should stay visible via contrast and the native platform outline.
- **Secondary / Ghost:** Secondary buttons use the control surface neutral with steel-blue borders. Destructive actions switch to the deep rose-danger surface.

### Metric Cards
- **Character:** Dense, signal-driven summaries that feel integrated with the shell.
- **Structure:** Tonal shell card with a small accent chip, muted label, and large numeric value.
- **Rule:** Never use a colored side stripe. If a metric needs emphasis, use the chip, the value, or the chart color role.

### Cards / Containers
- **Corner Style:** 10px to 12px on major panels, 8px on controls.
- **Background:** Shell Surface for primary panels, Shell Deep for nested control groups.
- **Border:** Steel Border, always present, never heavy.
- **Internal Padding:** 16px to 22px depending on density. Larger workspaces use 18px or 22px.

### Inputs / Fields
- **Style:** Input Bed background with Steel Border and 8px radius.
- **Focus:** Focus is contrast-led. Inputs stay dark but become visually clearer through the native focus ring and surrounding field contrast.
- **Error / Disabled:** Disabled fields dim toward faint text and muted borders rather than collapsing into low-contrast gray.

### Navigation
- **Top-Level:** The main shell uses grouped action clusters instead of a traditional sidebar.
- **Intra-Screen:** Camera switching is handled by a dedicated selector rail. Analytics uses tabs with the selected tab shifting to a brighter neutral state.
- **Map Toggles:** Coverage, overlap, and people visibility are treated as compact selection controls inside the map workspace, not global toolbar buttons.

### Signature Component
- **Camera Switcher Rail:** A stacked set of status-rich selector buttons. Each button exposes name, runtime state, FPS, and drift in one compact block. Selecting a camera updates both the primary feed and the map context.

## Do's and Don'ts

### Do:
- **Do** keep one camera feed visually dominant while the other feeds remain immediately reachable.
- **Do** use `#75d3e0` or `#7de3e1` only for active focus, live-state signaling, and primary action confirmation.
- **Do** pair the selected camera with its map coverage context so the operator never has to infer which footprint they are looking at.
- **Do** keep panel surfaces between `#071018` and `#132230`, with `#1f3141` as the default boundary stroke.
- **Do** preserve dense telemetry when it improves decisions, but render it with muted text and stable structure.

### Don't:
- **Don't** make this look like a generic purple SaaS dashboard.
- **Don't** turn the camera area into a CCTV wall with equal-weight tiles.
- **Don't** make it feel like a consumer smart-home control app.
- **Don't** use glossy marketing visuals, decorative charts, or identity-centric surveillance styling.
- **Don't** use colored `border-left` accents on cards, list items, or alerts. Emphasis belongs in full-surface state, chips, or typography.
