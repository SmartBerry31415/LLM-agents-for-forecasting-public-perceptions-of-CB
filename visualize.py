import json
import matplotlib.pyplot as plt
import numpy as np

# Load trust history data
with open("trust_history_depseek_r1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

plt.figure(figsize=(12, 8))
ax = plt.gca()

# Define trust regions and colors with labels
regions = [
    {"min": 0.0, "max": 0.5, "color": "lightcoral", "label": "Negative"},
    {"min": 0.5, "max": 0.6, "color": "lightyellow", "label": "Neutral"},
    {"min": 0.6, "max": 1.0, "color": "lightgreen", "label": "Positive"}
]

# Draw trust regions with labels
for region in regions:
    ax.axhspan(region["min"], region["max"], 
               facecolor=region["color"], 
               alpha=0.3)
    ax.axhline(y=region["min"], color="gray", linestyle="--", alpha=0.7)
    ax.axhline(y=region["max"], color="gray", linestyle="--", alpha=0.7)
    
    # Add region label
    ax.text(0.5, (region["min"] + region["max"])/2, region["label"],
            fontsize=12, ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
            transform=ax.get_yaxis_transform())

# Get distinct colors for agents
colors = plt.cm.tab10(np.linspace(0, 1, len(data)))

# Plot each agent's trust trajectory with solid lines
for (agent, agent_data), color in zip(data.items(), colors):
    trust_history = agent_data["trust_history"]
    n_questions = (len(trust_history) - 1) // 2  # Calculate number of questions
    
    # Extract trust value after each full question (both iterations)
    x = np.arange(1, n_questions + 1)
    y = [trust_history[2*i] for i in range(1, n_questions + 1)]
    
    # Plot with distinct color and label
    plt.plot(x, y, 'o-', linewidth=2, markersize=6, 
             label=agent, color=color, alpha=0.9)

plt.title("Agent Trust Dynamics Through Focus Group Discussion using DeepSeek-R1", fontsize=14)
plt.xlabel("Question Number", fontsize=12)
plt.ylabel("Trust Level", fontsize=12)
plt.xticks(range(1, n_questions + 1))
plt.ylim(0, 1)
plt.grid(alpha=0.2)

# Create custom legend
handles, labels = ax.get_legend_handles_labels()
legend = plt.legend(handles, labels, loc='upper center', 
                    bbox_to_anchor=(0.5, -0.15),
                    ncol=min(4, len(data)), 
                    fontsize=10)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make space for bottom legend
plt.savefig("trust_dynamics.png", dpi=300, bbox_inches='tight')
plt.show()