"""
Component-type-aware weights for context-sensitive layout analysis.

Each UI component type has different expectations for density, alignment, and overlap.
For example:
- Navigation items SHOULD cluster together (low density penalty)
- Buttons SHOULD align precisely (high alignment weight)
- Dialogs CAN overlay other content (low overlap penalty)

These weights scale the penalties in clutter and alignment calculations to reflect
real-world design patterns rather than treating all components identically.
"""

COMPONENT_WEIGHTS = {
    "heading": {
        "density_penalty": 1.0,      # Headings spread normally
        "alignment_weight": 1.3,     # Should align with grid
        "overlap_penalty": 1.3,      # Should not overlap
    },
    "link": {
        "density_penalty": 1.0,      # Links can cluster moderately
        "alignment_weight": 1.0,     # Neutral alignment
        "overlap_penalty": 1.0,      # Should not overlap
    },
    "image": {
        "density_penalty": 1.0,      # Images spread normally
        "alignment_weight": 0.7,     # Can break grid for visual impact
        "overlap_penalty": 1.0,      # Should not overlap
    },
    "text": {
        "density_penalty": 1.0,      # Text blocks spread normally
        "alignment_weight": 1.0,     # Neutral alignment
        "overlap_penalty": 1.2,      # Overlapping text is broken UI
    },
    "list": {
        "density_penalty": 0.5,      # List items SHOULD cluster (stacked vertically)
        "alignment_weight": 1.0,     # Neutral alignment
        "overlap_penalty": 1.0,      # Should not overlap
    },
    "header": {
        "density_penalty": 0.7,      # Header items can cluster
        "alignment_weight": 1.1,     # Should align moderately
        "overlap_penalty": 1.1,      # Should not overlap
    },
    "footer": {
        "density_penalty": 0.6,      # Footer items often cluster
        "alignment_weight": 1.1,     # Should align moderately
        "overlap_penalty": 1.1,      # Should not overlap
    },
    "table": {
        "density_penalty": 1.0,      # Tables spread normally
        "alignment_weight": 1.0,     # Neutral alignment
        "overlap_penalty": 1.2,      # Overlapping cells = broken table
    },
    "input": {
        "density_penalty": 1.2,      # Form fields need breathing room
        "alignment_weight": 1.5,     # MUST align (forms are grid-based)
        "overlap_penalty": 1.5,      # Overlapping inputs = broken form
    },
    "button": {
        "density_penalty": 1.2,      # Buttons need spacing
        "alignment_weight": 1.5,     # MUST align (CTA groups)
        "overlap_penalty": 1.5,      # Overlapping buttons = broken UI
    },
    "navigation": {
        "density_penalty": 0.5,      # Nav items SHOULD cluster (menu bars)
        "alignment_weight": 1.4,     # Nav items align horizontally/vertically
        "overlap_penalty": 0.7,      # Dropdowns can overlap main content
    },
    "sidebar": {
        "density_penalty": 0.5,      # Sidebar items stack vertically
        "alignment_weight": 0.8,     # Sidebar separate from main grid
        "overlap_penalty": 0.6,      # Can overlap main content edge
    },
    "dialog": {
        "density_penalty": 0.8,      # Dialogs moderate density
        "alignment_weight": 0.5,     # Dialogs don't align with page grid (overlays)
        "overlap_penalty": 0.2,      # SHOULD overlay page content (modals)
    },
    "container": {
        "density_penalty": 0.9,      # Containers moderate density
        "alignment_weight": 0.9,     # Neutral alignment
        "overlap_penalty": 0.9,      # Containers expected to nest
    },
}