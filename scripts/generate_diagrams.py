import os
from graphviz import Digraph

def generate_architecture_diagram():
    os.makedirs("diagrams", exist_ok=True)
    
    dot = Digraph(comment='Evaluation Architecture', format='png')
    dot.attr(rankdir='LR', size='12,6', dpi='300')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', margin='0.2')
    dot.attr('edge', fontname='Helvetica', fontsize='10', color='#666666')
    
    dot.node('Candidate', 'Candidate Output\n(System Under Test)', fillcolor='#f0f0f0', shape='component')
    
    with dot.subgraph(name='cluster_eval') as c:
        c.attr(label='Evaluation Panel', style='dashed', color='blue', bgcolor='#f9f9ff')
        c.node('LLMJudge', 'LLM Judge', fillcolor='#cce5ff')
        c.node('Symbolic', 'Symbolic Verifier', fillcolor='#e5ffcc')
        c.node('Prog', 'Programmatic Test', fillcolor='#ffe5cc')
        c.node('RAG', 'Retrieval Check', fillcolor='#ffcce5')
        
    dot.node('ECS', 'Compute ECS\n(Disagreement Metric)', fillcolor='#e6e6fa', shape='hexagon')
    dot.node('Decision', 'ECS > Threshold?', shape='diamond', fillcolor='#fffacd')
    dot.node('Escalate', 'Escalate / Add\nEvaluators', fillcolor='#ffb6c1')
    dot.node('Aggregate', 'Correlation-Aware\nAggregation', fillcolor='#98fb98')
    dot.node('Verdict', 'Final Verdict &\nConfidence Score', fillcolor='#ffd700', shape='Mrecord')
    
    dot.edge('Candidate', 'LLMJudge')
    dot.edge('LLMJudge', 'ECS')
    dot.edge('ECS', 'Decision')
    dot.edge('Decision', 'Aggregate', label='No (High Agreement)', color='green', fontcolor='green')
    dot.edge('Decision', 'Escalate', label='Yes (Disagreement)', color='red', fontcolor='red')
    dot.edge('Escalate', 'Symbolic')
    dot.edge('Escalate', 'Prog')
    dot.edge('Escalate', 'RAG')
    dot.edge('Symbolic', 'ECS')
    dot.edge('Prog', 'ECS')
    dot.edge('RAG', 'ECS')
    dot.edge('Escalate', 'Aggregate', label='Max Escalation', style='dotted')
    dot.edge('Aggregate', 'Verdict')
    
    dot.render('diagrams/placeholder_architecture', cleanup=True)
    print("Generated placeholder_architecture.png")

if __name__ == "__main__":
    generate_architecture_diagram()
