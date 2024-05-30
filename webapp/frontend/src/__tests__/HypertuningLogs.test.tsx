import { render } from '@testing-library/react';
import HypertuningLogs from '../logs_routes/HypertuningLogs';

test('renders HypertuningLogs component', () => {
    const { container } = render(<HypertuningLogs />);
    const iframe = container.querySelector('iframe');

    expect(iframe).toBeInTheDocument();
    expect(iframe).toHaveAttribute('src', '/hypertuning_logs.html');
});
