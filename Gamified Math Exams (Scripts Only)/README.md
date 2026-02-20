# Gamified Math Assessment System (Unity - Source Code)

This repository contains the core C# scripts and architectural logic for a gamified educational assessment platform developed in Unity. The system focuses on creating an immersive exam environment with interactive world objects and advanced accessibility features for mathematical content.

> **Note**: This is a **Source-Code Only** view. Due to the project's total size (12GB+), binary assets such as 3D models, textures, and high-resolution lightmaps have been excluded to keep the focus on the C# implementation and system design.

## 🌟 Key Features & Logic

### 1. Intelligent Math-to-Speech Parser (`TextReader.cs`)
Standard Text-to-Speech (TTS) engines often misinterpret mathematical symbols. This system includes a custom preprocessing layer that translates mathematical notation into natural, human-readable language before audio synthesis.
* **Symbol Translation**: Converts arithmetic operators like `*` to "multiplied" and `-` to "minus".
* **Contextual Logic**: Specifically identifies `??` placeholders in exam questions and reads them as "blank" to maintain the logical flow for the student.
* **Async Feedback**: Implements `SVSFlagsAsync` via the `SpeechLib` library to ensure audio feedback does not interrupt the main game thread or player movement.

### 2. Modular Interaction System (`StationaryObject.cs`)
Built on the principle of **Polymorphism**, the interaction system uses a base class to handle environmental triggers.
* **Scalability**: By using `public virtual void Interact()`, any object in the environment (from simple crates to complex terminals) can be added without modifying the player's core interaction logic.
* **Event-Driven UI**: Highlighting and interaction prompts are handled via C# Event Handlers in `StationaryObjectVisualInteractEnabled.cs`. This prevents the need for expensive distance-checking calculations in the `Update()` loop, optimizing performance for lower-end hardware.

### 3. Centralized Audio Management (`SoundManager.cs`)
A decoupled audio system that manages all gamification sounds (Success/Fail chimes, placement sounds, and UI feedback). This allows for easy global sound adjustments and ensures that audio triggers are not tied to individual world assets.

## 🛠 Tech Stack
* **Engine**: Unity 2022.3+
* **Language**: C#
* **Libraries**: `SpeechLib` (Windows COM-based TTS)
* **Design Patterns**: Singleton (Audio), Inheritance (Interactions), and Observer (Visual Events).

## 📂 Featured Scripts
* `TextReader.cs`: The core accessibility engine for mathematical reading.
* `StationaryObject.cs`: The base class for all environmental interactions.
* `StationaryObjectVisualInteractEnabled.cs`: Logic for high-performance visual feedback.
* `SoundManager.cs`: Centralized controller for gamified audio triggers.

## ⚖️ Disclaimer
This code is provided for portfolio review and educational purposes. It demonstrates proficiency in C#, object-oriented programming in Unity, and the development of accessible educational software.

---
*Developed by [Your Name/GitHub Username]*
