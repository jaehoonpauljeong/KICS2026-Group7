
from load_model import predict_with_explanation

import pandas as pd
from openai import OpenAI

def build_llm_prompt(result, top_k=10):
    
    feature_descriptions = {
    "duration": "Length of the network connection in seconds.",
    "protocol_type": "Network protocol used by the connection (e.g., TCP, UDP, ICMP).",
    "service": "Destination network service requested (e.g., HTTP, FTP, Telnet, SSH).",
    "flag": "Status flag of the connection based on the TCP state machine.",
    "src_bytes": "Number of data bytes sent from source to destination.",
    "dst_bytes": "Number of data bytes sent from destination to source.",
    "land": "1 if source and destination IP/port are identical; otherwise 0.",
    "wrong_fragment": "Number of incorrect or malformed IP fragments in the connection.",
    "urgent": "Number of urgent packets sent using the TCP urgent pointer.",
    "hot": "Count of suspicious system-level operations (e.g., file access, exec).",
    "num_failed_logins": "Number of failed login attempts associated with the connection.",
    "logged_in": "1 if the user successfully logged in; otherwise 0.",
    "num_compromised": "Count of signs indicating that the system may be compromised.",
    "root_shell": "1 if a root shell was obtained; otherwise 0.",
    "su_attempted": "1 if the 'su root' command was attempted; otherwise 0.",
    "num_root": "Number of operations performed as root.",
    "num_file_creations": "Number of file creation operations during the connection.",
    "num_shells": "Number of shell prompts invoked.",
    "num_access_files": "Number of attempts to access sensitive files.",
    "num_outbound_cmds": "Number of outbound commands issued (not used in KDD99; always 0).",
    "is_host_login": "1 if the login belongs to a host account; otherwise 0.",
    "is_guest_login": "1 if the login belongs to a guest account; otherwise 0.",
    "count": "Number of connections to the same host in the past 2 seconds.",
    "srv_count": "Number of connections to the same service in the past 2 seconds.",
    "serror_rate": "Percentage of connections with SYN errors to the same host.",
    "srv_serror_rate": "Percentage of connections with SYN errors to the same service.",
    "rerror_rate": "Percentage of connections with REJ errors to the same host.",
    "srv_rerror_rate": "Percentage of connections with REJ errors to the same service.",
    "same_srv_rate": "Percentage of connections to the same service.",
    "diff_srv_rate": "Percentage of connections to different services.",
    "srv_diff_host_rate": "Percentage of connections to the same service but different host.",
    "dst_host_count": "Number of connections to the destination host in the past 2 seconds.",
    "dst_host_srv_count": "Number of connections to the destination service in the past 2 seconds.",
    "dst_host_same_srv_rate": "Percentage of connections to the same service for the destination host.",
    "dst_host_diff_srv_rate": "Percentage of connections to different services for the destination host.",
    "dst_host_same_src_port_rate": "Percentage of connections from the same source port to the destination.",
    "dst_host_srv_diff_host_rate": "Percentage of connections to the same service but different hosts.",
    "dst_host_serror_rate": "Percentage of SYN-error connections to the destination host.",
    "dst_host_srv_serror_rate": "Percentage of SYN-error connections to the service on the host.",
    "dst_host_rerror_rate": "Percentage of REJ-error connections to the destination host.",
    "dst_host_srv_rerror_rate": "Percentage of REJ-error connections to the destination service.",
    "labels": "Attack class or label associated with the connection."
    }
    
    pred = result["prediction"]
    contribs = result["explanation"]["feature_contributions"]

    # Sort features by absolute SHAP importance
    sorted_features = sorted(
        contribs.values(), key=lambda x: abs(x["contribution"]), reverse=True
    )[:top_k]

    lines = []
    lines.append(f"IDS Prediction: {pred['predicted_class']}")
    lines.append(f"Confidence: {pred['confidence']:.3f}")
    lines.append("")
    lines.append("Class probabilities:")
    for cls, p in pred["all_probabilities"].items():
        lines.append(f"  - {cls}: {p:.3f}")
    lines.append("")
    lines.append("Top contributing features (SHAP values):")

    for f in sorted_features:
        lines.append(
            f"- {f['feature_name']}: value={f['value']}, SHAP={f['contribution']:.4f}"
        )
        if f['feature_name'] in feature_descriptions:
            lines.append(f"    Description: {feature_descriptions[f['feature_name']]}") 


    lines.append("")
    lines.append(
        "Using the information above, provide a clear and technically accurate explanation that includes:"
        "\n1) Why the model classified this network flow as the predicted attack type."
        "\n2) A detailed description of this attack type and how it typically behaves in network traffic."
        "\n3) The potential security impact or risk associated with this traffic pattern."
        "\n4) Specific mitigation recommendations for network defenders."
        "\n5) Any uncertainty or ambiguity present in the model’s decision, based on the SHAP contributions."
    )

    return "\n".join(lines)


def generate_explanation(prompt):

    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
    )
    
    response = client.chat.completions.create(
        model="deepseek/deepseek-v3.2-exp",
        messages=[
            {"role": "system", "content": "You are an expert explaining Intrusion Detection System outputs with SHAP."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
    )
    
    return response.choices[0].message.content.strip()



csv_file_path = "kdd_test.csv"
df = pd.read_csv(csv_file_path)

row = df.iloc[4]
input_data = row.to_dict()
result = predict_with_explanation(input_data)
prompt = build_llm_prompt(result, top_k=10)
print(prompt)
explanation = generate_explanation(prompt)
print("\n=== LLM Explanation ===\n")
print(explanation)
