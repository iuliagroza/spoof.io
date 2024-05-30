import { render } from '@testing-library/react';
import TrainingLogs from '../logs_routes/TrainingLogs';

test('renders TrainingLogs component', () => {
    const { container } = render(<TrainingLogs />);
    const iframe = container.querySelector('iframe');

    expect(iframe).toBeInTheDocument();
    expect(iframe).toHaveAttribute('src', '/test_results.html');
});
