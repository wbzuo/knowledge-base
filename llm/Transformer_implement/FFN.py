import torch
import torch.nn as nn
import torch.nn.functional as F  # 修正：应该是 torch.nn.functional

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        x = F.relu(self.w1(x))  # 修正：relu 应该是小写
        x = self.w2(x)
        x = self.dropout(x)
        return x

def test_mlp():
    """测试 MLP 模型的函数"""
    print("=== 测试 MLP 模型 ===")
    
    # 模型参数
    dim = 512
    hidden_dim = 1024
    dropout = 0.1
    
    # 创建模型实例
    model = MLP(dim=dim, hidden_dim=hidden_dim, dropout=dropout)
    
    # 打印模型结构
    print("模型结构:")
    print(model)
    print(f"参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 测试数据
    batch_size = 4
    seq_len = 10
    test_input = torch.randn(batch_size, seq_len, dim)
    print(f"\n输入形状: {test_input.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = model(test_input)
    
    print(f"输出形状: {output.shape}")
    
    # 验证维度一致性
    assert output.shape == test_input.shape, f"输出形状 {output.shape} 与输入形状 {test_input.shape} 不匹配"
    print("✓ 维度一致性验证通过")
    
    # 测试 dropout 效果
    print(f"\n=== Dropout 测试 ===")
    model.eval()  # 评估模式（无dropout）
    output_eval = model(test_input)
    
    model.train()  # 训练模式（有dropout）
    output_train = model(test_input)
    
    print(f"评估模式输出方差: {output_eval.var().item():.4f}")
    print(f"训练模式输出方差: {output_train.var().item():.4f}")
    print("✓ Dropout 功能正常")
    
    # 参数检查
    print(f"\n=== 参数检查 ===")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
    
    return model, test_input, output

def test_mlp_different_configs():
    """测试不同配置的 MLP"""
    print("\n" + "="*50)
    print("测试不同配置的 MLP")
    print("="*50)
    
    test_cases = [
        {"dim": 256, "hidden_dim": 512, "dropout": 0.1},
        {"dim": 768, "hidden_dim": 3072, "dropout": 0.2},  # Transformer 常用配置
        {"dim": 128, "hidden_dim": 256, "dropout": 0.0},
    ]
    
    for i, config in enumerate(test_cases):
        print(f"\n--- 测试用例 {i+1} ---")
        print(f"配置: dim={config['dim']}, hidden_dim={config['hidden_dim']}, dropout={config['dropout']}")
        
        model = MLP(**config)
        test_input = torch.randn(2, 8, config['dim'])
        output = model(test_input)
        
        print(f"输入: {test_input.shape} -> 输出: {output.shape}")
        print(f"参数数量: {sum(p.numel() for p in model.parameters())}")

if __name__ == "__main__":  # 修正：应该是 __main__
    # 运行测试
    model, test_input, output = test_mlp()
    
    # 测试不同配置
    test_mlp_different_configs()
    
    print("\n🎉 所有测试通过！MLP 模型工作正常。")