print(f"\nAnalyzing: {best_model_name}")

tn = best_metrics['tn']
fp = best_metrics['fp']
fn = best_metrics['fn']
tp = best_metrics['tp']

print(f"\nConfusion Matrix Components:")
print(f"  True Negatives (Benign correctly identified):      {tn:3d}")
print(f"  False Positives (Benign misclassified):            {fp:3d}")
print(f"  False Negatives (Malignant misclassified):         {fn:3d}  [CRITICAL]")
print(f"  True Positives (Malignant correctly identified):   {tp:3d}")

# Calculate error rates
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\nClinical Metrics:")
print(f"  Sensitivity (True Positive Rate):  {sensitivity:.4f} ({sensitivity*100:.2f}%)")
print(f"  Specificity (True Negative Rate):  {specificity:.4f} ({specificity*100:.2f}%)")
print(f"  False Negative Rate:               {fnr:.4f} ({fnr*100:.2f}%)")
print(f"  False Positive Rate:               {fpr:.4f} ({fpr*100:.2f}%)")

print("\nMEDICAL IMPLICATIONS")

print(f"\n FALSE NEGATIVES ({fn} cases): MOST CRITICAL")
print("   Impact:")
print("   • Malignant tumors classified as benign")
print("   • Delayed diagnosis and treatment")
print("   • Potentially life-threatening consequences")
print("   • Reduced patient survival rates")
print("   • Legal and ethical implications")

print(f"\n FALSE POSITIVES ({fp} cases): LESS SEVERE")
print("   Impact:")
print("   • Benign tumors classified as malignant")
print("   • Unnecessary anxiety for patients")
print("   • Additional diagnostic procedures (biopsies)")
print("   • Increased healthcare costs")
print("   • Patient psychological distress")

print(f"\n CLINICAL RECOMMENDATIONS:")
if fnr < 0.03:
    print("   Excellent: FN rate < 3% is clinically acceptable")
elif fnr < 0.05:
    print("   Good: FN rate < 5% is within acceptable range")
else:
    print("   Concerning: FN rate > 5% may require threshold adjustment")

print(f"\n DEPLOYMENT STRATEGY:")
print("1. Use as first-line screening tool, not final diagnosis")
print("2. All positive predictions require clinical confirmation")
print("3. Borderline cases (confidence < 80%) flagged for review")
print("4. Combine with radiologist expert judgment")
print("5. Regular model performance monitoring and retraining")
