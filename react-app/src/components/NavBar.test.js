import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import NavBar from './NavBar';

describe('NavBar Component', () => {
  test('renders with initial active menu item as "Logs"', () => {
    const mockMenuChange = jest.fn();
    render(<NavBar onMenuChange={mockMenuChange} />);

    // Expect "Logs" to be active initially
    const logsItem = screen.getByText('Logs');
    expect(logsItem).toHaveClass('active');

    // Other items should not be active
    expect(screen.getByText('Results')).not.toHaveClass('active');
    expect(screen.getByText('Sample')).not.toHaveClass('active');
    expect(screen.getByText('Parameters')).not.toHaveClass('active');
  });

  test('changes active item on click', () => {
    const mockMenuChange = jest.fn();
    render(<NavBar onMenuChange={mockMenuChange} />);

    // Click on "Results"
    const resultsItem = screen.getByText('Results');
    fireEvent.click(resultsItem);

    // Expect "Results" to become active
    expect(resultsItem).toHaveClass('active');
    expect(screen.getByText('Logs')).not.toHaveClass('active');
    // Check if onMenuChange was called correctly
    expect(mockMenuChange).toHaveBeenCalledWith('Results');
  });

  
});
