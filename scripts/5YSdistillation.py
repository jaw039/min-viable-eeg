"""
Knowledge distillation for EEG channel reduction with Q5 falsifiability controls.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from src.models import EEGNet

class DistillationLoss(nn.Module):
    """Combined loss for knowledge distillation."""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits: torch.Tensor, 
                teacher_logits: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined distillation loss.
        
        Args:
            student_logits: Output from student model
            teacher_logits: Output from teacher model
            targets: Ground truth labels
        
        Returns:
            Combined loss
        """
        # Hard loss (cross-entropy on true labels)
        hard_loss = self.ce_loss(student_logits, targets)
        
        # Soft loss (KL divergence on soft targets)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        
        # Combined loss
        return self.alpha * hard_loss + (1 - self.alpha) * self.temperature**2 * soft_loss

class ShuffledTeacher:
    """Q5 Control 1: Teacher with shuffled predictions."""
    
    def __init__(self, teacher: nn.Module, shuffle_seed: int = 42):
        self.teacher = teacher
        self.shuffle_seed = shuffle_seed
    
    @torch.no_grad()
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        logits = self.teacher(X)
        # Shuffle batch predictions
        np.random.seed(self.shuffle_seed)
        batch_size = logits.shape[0]
        perm = torch.from_numpy(np.random.permutation(batch_size))
        return logits[perm]

class UntrainedTeacher:
    """Q5 Control 2: Untrained teacher with random weights."""
    
    def __init__(self, n_channels: int, n_classes: int, n_samples: int):
        self.model = EEGNet(
            n_channels=n_channels,
            n_classes=n_classes,
            n_samples=n_samples
        )
        # Random initialization (no training)
    
    @torch.no_grad()
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)

class DistillationController:
    """Manages distillation experiments with Q5 controls."""
    
    def __init__(self, teacher: nn.Module, student: nn.Module,
                 temperature: float = 4.0, alpha: float = 0.7):
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        
        # Ensure teacher is in eval mode
        self.teacher.eval()
    
    def train_distilled(self, train_loader, val_loader, epochs: int = 100) -> nn.Module:
        """Standard distillation training."""
        optimizer = torch.optim.Adam(self.student.parameters(), lr=1e-3)
        criterion = DistillationLoss(self.temperature, self.alpha)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.student.train()
            train_loss = 0
            
            for X_reduced, X_full, y in train_loader:
                optimizer.zero_grad()
                
                # Teacher output (full channels)
                with torch.no_grad():
                    teacher_logits = self.teacher(X_full)
                
                # Student output (reduced channels)
                student_logits = self.student(X_reduced)
                
                loss = criterion(student_logits, teacher_logits, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.student.eval()
            val_loss = 0
            with torch.no_grad():
                for X_reduced, X_full, y in val_loader:
                    teacher_logits = self.teacher(X_full)
                    student_logits = self.student(X_reduced)
                    loss = criterion(student_logits, teacher_logits, y)
                    val_loss += loss.item()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                best_state = self.student.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break
        
        # Restore best model
        self.student.load_state_dict(best_state)
        return self.student
    
    def train_shuffled_teacher(self, train_loader, val_loader, epochs: int = 100) -> nn.Module:
        """
        Q5 Control 1: Train with teacher that has shuffled predictions.
        If this helps, effect is regularization, not knowledge transfer.
        """
        shuffled_teacher = ShuffledTeacher(self.teacher)
        return self._train_with_custom_teacher(train_loader, val_loader, 
                                               shuffled_teacher, epochs)
    
    def train_untrained_teacher(self, train_loader, val_loader, 
                                full_channels: int, n_classes: int, 
                                n_samples: int, epochs: int = 100) -> nn.Module:
        """
        Q5 Control 2: Train with untrained teacher.
        If this helps, effect is regularization.
        """
        untrained_teacher = UntrainedTeacher(full_channels, n_classes, n_samples)
        return self._train_with_custom_teacher(train_loader, val_loader, 
                                               untrained_teacher, epochs)
    
    def _train_with_custom_teacher(self, train_loader, val_loader, 
                                   teacher, epochs: int = 100) -> nn.Module:
        """Internal training with arbitrary teacher."""
        optimizer = torch.optim.Adam(self.student.parameters(), lr=1e-3)
        criterion = DistillationLoss(self.temperature, self.alpha)
        
        for epoch in range(epochs):
            self.student.train()
            for X_reduced, X_full, y in train_loader:
                optimizer.zero_grad()
                
                with torch.no_grad():
                    teacher_logits = teacher(X_full)
                
                student_logits = self.student(X_reduced)
                loss = criterion(student_logits, teacher_logits, y)
                loss.backward()
                optimizer.step()
        
        return self.student

def get_distillation_description() -> str:
    """Return description for methods section."""
    return """
    Knowledge distillation was evaluated with two control conditions (Q5):
    1. Shuffled teacher: teacher predictions were shuffled at the batch level
    2. Untrained teacher: teacher weights were random (no training)
    
    If either control matches the real teacher's performance, the distillation
    benefit is attributed to regularization rather than knowledge transfer.
    """