"use client";

import { PanelLeftOpen, Sparkles } from "lucide-react";

interface HeaderProps {
  onMenuClick: () => void;
  /** Whether the sidebar is collapsed on desktop */
  sidebarCollapsed: boolean;
}

export default function Header({ onMenuClick, sidebarCollapsed }: HeaderProps) {
  return (
    <header className="sticky top-0 z-30 border-b border-cream-200/70 bg-cream-50/80 backdrop-blur-lg">
      <div className="flex items-center justify-between px-4 h-14">
        {/* Left: Menu + Title */}
        <div className="flex items-center gap-3">
          {/* 
            Mobile: always show the button (sidebar is overlay-based)
            Desktop: only show when sidebar is collapsed 
          */}
          <button
            onClick={onMenuClick}
            className={`
              p-2 rounded-xl
              text-brand-600 hover:bg-cream-200/60
              transition-all duration-200
              ${sidebarCollapsed ? "lg:flex" : "lg:hidden"}
            `}
            title="Open sidebar"
          >
            <PanelLeftOpen size={20} />
          </button>

          <div className="flex items-center gap-2">
            <div className="hidden sm:flex w-7 h-7 rounded-lg bg-gradient-to-br from-brand-400 to-brand-600 items-center justify-center shadow-warm">
              <Sparkles size={14} className="text-white" />
            </div>
            <h1 className="text-base font-bold text-brand-800 tracking-tight">
              CAFS OnlineCE Assistant
            </h1>
          </div>
        </div>

        {/* Right: spacer */}
        <div />
      </div>
    </header>
  );
}
