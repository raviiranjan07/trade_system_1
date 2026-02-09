import React from 'react'

function TabBar({ activeTab, onTabChange }) {
  const tabs = [
    { id: 'trades', label: 'Trades' },
    { id: 'signals', label: 'Signals' },
  ]

  return (
    <nav className="tab-bar">
      {tabs.map(tab => (
        <button
          key={tab.id}
          className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
          onClick={() => onTabChange(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </nav>
  )
}

export default TabBar
