import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_SCRIPT_TIMEOUT = 60


@dataclass
class SkillMetadata:
    """Represents an Agent Skill with metadata parsed from SKILL.md frontmatter.

    see: AgentSkills Specification: https://agentskills.io/specification

    """

    name: str
    description: str
    path: str  # Absolute path to the skill directory
    license: Optional[str] = None
    compatibility: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    allowed_tools: Optional[str] = None


def parse_skill_frontmatter(skill_md_path: str) -> Optional[SkillMetadata]:
    """
    Parse the YAML frontmatter from a SKILL.md file.
    Returns a Skill object with metadata, or None if parsing fails.
    """
    try:
        with open(skill_md_path, mode="r", encoding="utf-8") as f:
            content = f.read()

        frontmatter_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if not frontmatter_match:
            return None

        frontmatter_yaml = frontmatter_match.group(1)
        frontmatter = yaml.safe_load(frontmatter_yaml)

        if not frontmatter:
            return None

        # Required fields
        name = frontmatter.get("name")
        description = frontmatter.get("description")

        if not name or not description:
            return None

        # Validate name format (lowercase, hyphens, no consecutive hyphens)
        if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", name):
            return None

        skill_dir = str(Path(skill_md_path).parent)

        return SkillMetadata(
            name=name,
            description=description,
            path=skill_dir,
            license=frontmatter.get("license"),
            compatibility=frontmatter.get("compatibility"),
            metadata=frontmatter.get("metadata", {}),
            # allowed_tools=frontmatter.get("allowed-tools"),
        )

    except Exception as e:
        logger.warning(f"Failed to parse skill at {skill_md_path}: {e}")
        return None


def discover_skills(skill_directories: List[str]) -> List[SkillMetadata]:
    """
    Discover skills from configured directories.
    A skill is a folder containing a SKILL.md file.

    Args:
        skill_directories: List of directory paths to scan for skills

    Returns:
        List of discovered Skill objects with parsed metadata
    """
    skills = []

    for skill_root_dir in skill_directories:
        skill_path = Path(skill_root_dir)

        if not skill_path.exists() or not skill_path.is_dir():
            logger.warning(
                f"Skill root directory does not exist or is not a directory, skipping: {skill_root_dir}"
            )
            continue
        for skill_dir in skill_path.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            skill = parse_skill_frontmatter(str(skill_md))
            if skill:
                skills.append(skill)
    return skills


def skills_to_xml(skills: List[SkillMetadata]) -> str:
    """
    Generate XML representation of available skills for system prompt injection.
    Follows the agentskills specification format.

    Args:
        skills: List of Skill objects

    Returns:
        XML string for inclusion in system prompt
    """
    if not skills:
        return ""

    lines = ["<available_skills>"]
    for skill in skills:
        lines.append("  <skill>")
        lines.append(f"    <name>{skill.name}</name>")
        # Escape XML special characters in description
        escaped_desc = (
            skill.description.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        lines.append(f"    <description>{escaped_desc}</description>")
        skill_md_path = str(Path(skill.path) / "SKILL.md")
        lines.append(f"    <location>{skill_md_path}</location>")
        lines.append("  </skill>")
    lines.append("</available_skills>")

    return "\n".join(lines)


def build_skills_system_prompt(skills: List[SkillMetadata]) -> str:
    """
    Build the system prompt section for skills.

    Args:
        skills: List of available Skill objects

    Returns:
        System prompt text including skill instructions and available skills XML
    """
    if not skills:
        return ""

    skills_xml = skills_to_xml(skills)

    return f"""
<agent_skills>
When users ask you to perform tasks, check if any of the available skills below can help complete the task more effectively. Skills provide specialized capabilities and domain knowledge.

To use a skill:
1. Read the skill's `SKILL.md` file (with the `load_skill_file` tool) at the provided location to get full instructions.
2. Follow the instructions within the skill to complete the task
3. Skills may include scripts, references, and assets that you can access as needed
4. Execute the script with the `execute_script` tool when the skill requires it

note:
- You can use the `load_skill_file` tool to load the file content (`SKILL.md` or other files in the skill directory).
- Only use skills when they are relevant to the current task.
- Not load same file (SKILL.md or other references file ) multiple times with `load_skill_file` tool.
- **Must not use the skills as tools, only use the tools provided by the model.**

{skills_xml}
</agent_skills>
"""


class SkillIntegrationTools:
    """
    Provide tools that used for skill integration, including:

    - load_skill_file: Load a file from a skill directory, such as `SKILL.md` or other files in the skill directory.
    - execute_script: Execute a script provided by the skill, such as `python scripts/now.py`.
    """

    def __init__(self, skills: List[SkillMetadata]) -> None:
        self.skills = {skill.name: skill for skill in skills}

    def load_skill_file(self, skill_name: str, file_path: str = "SKILL.md") -> str:
        """
        Load a file from a skill directory.

        Args:
            skill_name (str): The name of the skill, must be one of the available skills.
            file_path (str): The path to the file to load, default is `SKILL.md`

        Returns:
            str: The content of the file or an error message.
        """
        try:
            skill_root = Path(self.skills[skill_name].path).resolve()
            target_path = (skill_root / file_path).resolve()

            if not target_path.is_relative_to(skill_root):
                return f"Error: Access denied. File '{file_path}' is outside the skill directory."

            if not target_path.exists():
                return f"Error: File '{file_path}' not found in skill '{skill_name}'."

            with open(target_path, "r", encoding="utf-8") as f:
                return f.read()

        except Exception as e:
            logger.warning(
                f"Failed to load file {file_path} from skill {skill_name}: {str(e)}",
                exc_info=True,
            )
            return f"Error: Failed to load file {file_path} from skill {skill_name}: {str(e)}"

    def execute_script(self, skill_name: str, command: str) -> str:
        """
        Execute a script provided by the skill (synchronous version).

        Args:
            skill_name (str): The name of the skill.
            command (str): The shell command to execute.

        Returns:
            str: The stdout and stderr of the script execution.
        """
        try:
            skill_root = Path(self.skills[skill_name].path).resolve()

            try:
                completed = subprocess.run(
                    command,
                    shell=True,
                    cwd=skill_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=DEFAULT_SCRIPT_TIMEOUT,
                    text=True,
                    errors="replace",
                )
            except subprocess.TimeoutExpired:
                return "<stdout></stdout><stderr>Error: Execution timed out.</stderr>"

            stdout_str = (completed.stdout or "").strip()
            stderr_str = (completed.stderr or "").strip()

            logger.info(f"Executed script {command} in skill {skill_name} successfully")

            return (
                f"<stdout>\n{stdout_str}\n</stdout>\n"
                f"<stderr>\n{stderr_str}\n</stderr>"
            )

        except Exception as e:
            logger.warning(
                f"Failed to execute script {command} in skill {skill_name}: {str(e)}",
                exc_info=True,
            )
            return f"<stdout></stdout><stderr>System Error: {str(e)}</stderr>"
