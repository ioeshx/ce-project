from prompt_templates import CIFAR100_classes

CIFAR50_classes = CIFAR100_classes[:50]
print("CIFAR50_classes = [")
for class_name in CIFAR50_classes:
	print(f" {class_name!r},")
print("]")
