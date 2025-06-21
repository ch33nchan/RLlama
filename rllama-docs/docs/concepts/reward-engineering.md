
## Step 8: Create a Flowchart Page

Let's add a page with flowcharts to explain the RLlama architecture:

```markdown
// docs/concepts/reward-engineering.md
---
id: reward-engineering
title: The Reward Engineering Problem
sidebar_label: Reward Engineering
slug: /concepts/reward-engineering
---

# The Reward Engineering Problem

Reward engineering is the process of designing reward functions that guide reinforcement learning agents toward desired behaviors. It's one of the most challenging aspects of reinforcement learning.

## Traditional Approach vs RLlama Approach

<div className="flowchart-container">
  <div className="flowchart">
    <h3>Traditional Approach</h3>
    <pre className="mermaid">
    graph TD
      A[Environment State] --> B[Monolithic Reward Function]
      B --> C[Single Reward Value]
      C --> D[Agent]
      D --> A
    </pre>
  </div>
  <div className="flowchart">
    <h3>RLlama Approach</h3>
    <pre className="mermaid">
    graph TD
      A[Environment State] --> B[Context Object]
      B --> C1[Reward Component 1]
      B --> C2[Reward Component 2]
      B --> C3[Reward Component 3]
      C1 --> D[Reward Engine]
      C2 --> D
      C3 --> D
      D --> E[Weighted Sum]
      E --> F[Agent]
      F --> A
    </pre>
  </div>
</div>

## RLlama Reward Calculation Process

<pre className="mermaid">
flowchart LR
    A[Environment Step] --> B[Create Context]
    B --> C[Compute Component Rewards]
    C --> D[Apply Component Weights]
    D --> E[Sum Weighted Rewards]
    E --> F[Return Total Reward]
    F --> G[Update Agent]
    G --> A
</pre>

## Component Lifecycle

<pre className="mermaid">
stateDiagram-v2
    [*] --> Initialization
    Initialization --> Compute
    Compute --> Compute: Next step
    Compute --> Reset: Episode end
    Reset --> Compute: New episode
    Reset --> [*]: Teardown
</pre>
