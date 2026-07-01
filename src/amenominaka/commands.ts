// amenominaka — clean-room Kit command stack (omni.kit.commands).
//
// Mirrors `omni.kit.commands`: undoable Command objects + a CommandStack with
// undo / redo and a bounded history. Each Command captures its inverse so
// `undo()` reverses `do()`. Executing a new command clears the redo stack
// (standard linear undo history — no branching).
//
// ADR-2605261800 §D6 / D10.4 amenominaka.

/** Abstract undoable command. Subclasses implement `do()` + `undo()`. */
export abstract class Command {
  name = "";
  abstract do(): void;
  abstract undo(): void;
}

/** Set a key on a dict-like target; undo restores the prior value (or removes
 *  the key if it was absent). */
export class SetAttributeCommand<T extends Record<string, unknown>> extends Command {
  name = "SetAttribute";
  private hadKey = false;
  private prevValue: unknown = undefined;

  constructor(
    private readonly target: T,
    private readonly key: keyof T & string,
    private readonly value: unknown,
  ) {
    super();
  }

  do(): void {
    this.hadKey = this.key in this.target;
    this.prevValue = this.target[this.key];
    (this.target as Record<string, unknown>)[this.key] = this.value;
  }
  undo(): void {
    if (this.hadKey) (this.target as Record<string, unknown>)[this.key] = this.prevValue;
    else delete (this.target as Record<string, unknown>)[this.key];
  }
}

export class CommandStack {
  private undoStack: Command[] = [];
  private redoStack: Command[] = [];

  constructor(private readonly historySize = 1000) {}

  /** Execute a command and push it onto the undo stack. An exception in
   *  `do()` is rethrown and the command is NOT pushed. Clears the redo stack. */
  execute(cmd: Command): void {
    cmd.do();
    this.undoStack.push(cmd);
    this.redoStack = [];
    if (this.undoStack.length > this.historySize) this.undoStack.shift();
  }

  /** Undo the most recent command (no-op if empty). Returns it, or null. */
  undo(): Command | null {
    const cmd = this.undoStack.pop();
    if (!cmd) return null;
    cmd.undo();
    this.redoStack.push(cmd);
    return cmd;
  }

  /** Redo the most recently undone command (no-op if empty). */
  redo(): Command | null {
    const cmd = this.redoStack.pop();
    if (!cmd) return null;
    cmd.do();
    this.undoStack.push(cmd);
    return cmd;
  }

  canUndo(): boolean {
    return this.undoStack.length > 0;
  }
  canRedo(): boolean {
    return this.redoStack.length > 0;
  }
  history(): string[] {
    return this.undoStack.map((c) => c.name || c.constructor.name);
  }
  clear(): void {
    this.undoStack = [];
    this.redoStack = [];
  }
}

// ── global command stack (omni.kit.commands.execute/undo/redo) ───────────────

let _globalStack: CommandStack | null = null;

function globalStack(): CommandStack {
  if (_globalStack === null) _globalStack = new CommandStack();
  return _globalStack;
}

export function execute(cmd: Command): void {
  globalStack().execute(cmd);
}
export function undo(): Command | null {
  return globalStack().undo();
}
export function redo(): Command | null {
  return globalStack().redo();
}
/** Reset the global command stack (test helper). */
export function _resetStack(): void {
  _globalStack = null;
}
