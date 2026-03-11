import torch
import torch.nn as nn


class DRRN(nn.Module):
	def __init__(self):
		super(DRRN, self).__init__()
		self.input = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=127, stride=1, padding=127 // 2, bias=False)
		self.conv1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=31, stride=1, padding=31 // 2, bias=False)
		self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=31, stride=1, padding=31 // 2, bias=False)
		self.output = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=31, stride=1, padding=31 // 2, bias=False)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		residual = x
		inputs = self.input(self.relu(x))
		out = inputs
		for _ in range(4):
			out = self.conv2(self.relu(self.conv1(self.relu(out))))
			out = torch.add(out, inputs)
		out = self.output(self.relu(out))
		out = torch.add(out, residual)
		return out
